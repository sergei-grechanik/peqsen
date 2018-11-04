
import numpy as np
import itertools
import collections
import attr
import hypothesis
from hypothesis import strategies

import peqsen.util
#from peqsen.util import printind
printind = lambda *_: None

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term

class SimpleEvaluator(Listener):
    """This listener immediately evaluates new hyperedges as they are added into the hypergraph.
    Note that this evaluator is very simple: it does not support recursive-only definitions and
    higher-order values.
    """

    def __init__(self, hypergraph, evaluate, generate, num_models=1):
        if hypergraph:
            hypergraph.listeners.add(self)

        # Node ~> [Value]
        self._node_to_values = {}

        self._num_models = num_models
        self.evaluate = evaluate
        self.generate = generate

    def add_models(count=1, generate=None):
        if generate is None:
            generate = self.generate

        # We do this by simply creating another evaluator and adding all the nodes to it
        se = SimpleEvaluator(None, evaluate=self.evaluate, generate=generate, num_models=count)
        se.on_add(None, self._node_to_values.keys())

        for node, vals in se._node_to_values.items():
            self._node_to_values[node].extend(vals)

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Node) and not e.outgoing:
                self._node_to_values[e] = [self.generate(e, i) for i in range(self._num_models)]

        evaluated_hyperedges = set()

        def _eval(e, evaluated_hyperedges=evaluated_hyperedges):
            if isinstance(e, Node):
                if not e in self._node_to_values:
                    found_unevaluated = False
                    for h in e.outgoing:
                        if h not in evaluated_hyperedges:
                            found_unevaluated = True
                            _eval(h)
                    if not found_unevaluated:
                        raise RuntimeError("Cannot evaluate node {}. Probably it has no "
                                           "non-recursive definitions.".format(e))
            else:
                if not e in evaluated_hyperedges:
                    evaluated_hyperedges.add(h)
                    dst_values = []
                    for d in e.dst:
                        if not d in self._node_to_values:
                            _eval(d)
                        dst_values.append(self._node_to_values[d])

                    computed = [self.evaluate(e.label, vals) for vals in zip(*dst_values)]

                    if e.src not in self._node_to_values:
                        self._node_to_values[e.src] = computed
                    else:
                        reference_values = self._node_to_values[e.src]
                        for i in range(self._num_models):
                            if reference_values[i] != computed[i]:
                                raise RuntimeError("Hyperedge {} evaluated on {} model to a value"
                                                   "\n{}\n but expected\n{}"
                                                   .format(e, computed[i], reference_values[i]))

        for e in elements:
            _eval(e)

    def on_merge(self, hypergraph, node, removed, added):
        pass

    def on_remove(self, hypergraph, elements):
        pass

    def on_remove_node(self, hypergraph, node, hyperedges):
        self._node_to_values.pop(node, None)


FiniteFunction = attr.make_class('FiniteFunction',
                                 ['name', 'arity', 'carrier_size', 'table', 'func'],
                                 frozen=True)
EvaluableSignature = attr.make_class('EvaluableSignature',
                                     ['signature', 'evaluate', 'generate', 'functions'],
                                     frozen=True)

@strategies.composite
def gen_finite_function(draw, num_elements=(1, 5), arity=(0, 3), name_len=5):
    num_elements = draw(gen_number(num_elements))
    arity = draw(gen_number(arity))
    table = draw(hypothesis.extra.numpy.arrays('int32', [num_elements] * arity,
                                               strategies.integers(0, num_elements - 1)))
    name = "F" + some_name(hash(table.tostring()), name_len)

    def func(*args, arity=arity, table=table):
        assert len(args) == arity
        return table[tuple(args)]

    return FiniteFunction(name=name, arity=arity, carrier_size=num_elements, table=table, func=func)

@strategies.composite
def gen_finite_evaluable_signature(draw, num_elements=(1, 5),
                                   num_functions=[(1,3), (0,3), (0,10), (0,1)]):
    """Generate an instance of EvaluableSignature.

    num_elements
        the number (or a range) of elements of the carrier set.

    num_functions
        a list of ranges, indicating how many functions of different arities to generate
    """
    elems = draw(gen_number(num_elements))

    funcs = []
    for arity, num in enumerate(num_functions):
        f = draw(gen_list(gen_finite_function(elems, arity), num))
        funcs.extend(f)

    func_dict = {(f.name, f.arity): f.func for f in funcs}
    signature = [(f.name, f.arity) for f in funcs]

    def _evaluate(label, args, func_dict=func_dict):
        return func_dict[(label, len(args))](*args)

    def _generate(node, idx, draw=draw, elems=elems):
        return draw(strategies.integers(0, elems - 1))

    return EvaluableSignature(signature=signature, evaluate=_evaluate,
                              generate=_generate, functions=funcs)

