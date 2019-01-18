
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

    def __init__(self, hypergraph, signature=None, data=None,
                 evaluate=None, generate=None, num_models=1):
        if signature is not None:
            if evaluate is None:
                evaluate = lambda lbl, vals, sig=signature: sig.evaluate[lbl](*vals)
            if generate is None:
                generate = lambda e, i, data=data, sig=signature: data.draw(sig.generate)

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

        self._num_models += count

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
                    evaluated_hyperedges.add(e)
                    dst_values = []
                    for d in e.dst:
                        if not d in self._node_to_values:
                            _eval(d)
                        d_eval = self._node_to_values[d]
                        assert len(d_eval) == self._num_models
                        dst_values.append(d_eval)

                    if dst_values:
                        computed = [self.evaluate(e.label, vals) for vals in zip(*dst_values)]
                    else:
                        computed = [self.evaluate(e.label, [])] * self._num_models
                    assert len(computed) == self._num_models

                    if e.src not in self._node_to_values:
                        self._node_to_values[e.src] = computed
                    else:
                        reference_values = self._node_to_values[e.src]
                        for i in range(self._num_models):
                            if reference_values[i] != computed[i]:
                                print(hypergraph)
                                for node, vs in self._node_to_values.items():
                                    print("{}[{}] = {}".format(node, i, vs[i]))
                                raise RuntimeError("Hyperedge {} evaluated on {}-th model to "
                                                   "a value:\n{}\nbut expected:\n{}"
                                                   .format(e, i, computed[i],
                                                           reference_values[i]))

        for e in elements:
            _eval(e)

    def on_merge(self, hypergraph, node, removed, added, reason):
        self.on_add(hypergraph, added)

    def on_remove(self, hypergraph, elements):
        pass

    def on_remove_node(self, hypergraph, node, hyperedges):
        self._node_to_values.pop(node, None)


