
import hypothesis
from hypothesis import strategies

import peqsen.util
from peqsen import Node, Hyperedge, Hypergraph, Term, gen_list, parse, make_equality

import inspect
import attr

FiniteFunction = attr.make_class('FiniteFunction',
                                 ['name', 'arity', 'carrier_size', 'table', 'func'],
                                 frozen=True)
EvaluableSignature = attr.make_class('EvaluableSignature',
                                     ['signature', 'evaluate', 'generate'],
                                     frozen=True)

@strategies.composite
def gen_finite_function(draw, num_elements=(1, 5), arity=(0, 3), name_len=5):
    num_elements = draw(gen_number(num_elements))
    arity = draw(gen_number(arity))
    table = draw(hypothesis.extra.numpy.arrays('int32', [num_elements] * arity,
                                               strategies.integers(0, num_elements - 1)))

    idx = GloballyIndexed()._global_index
    name = "F" + some_name(hash(table.tostring()), name_len) + str(idx)

    def func(*args, arity=arity, table=table):
        assert len(args) == arity
        return table[tuple(args)]

    return FiniteFunction(name=name, arity=arity, carrier_size=num_elements, table=table, func=func)

@strategies.composite
def gen_finite_evaluable_signature(draw, num_elements=(1, 5),
                                   num_functions=[(1,3), (0,3), (0,10), (0,2)]):
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

    evaluate = {f.name: f.func for f in funcs}
    signature = {f.name: f.arity for f in funcs}

    def _generate(node, idx, draw=draw, elems=elems):
        return draw(strategies.integers(0, elems - 1))

    return EvaluableSignature(signature=signature, evaluate=evaluate, generate=_generate)

Theory = attr.make_class('Theory', ['signature', 'equalities'], frozen=True)

def make_evaluable_sig(labels2funs, generate):
    signature = {}
    evaluate = {}
    for l, f in labels2funs.items():
        if callable(f):
            signature[l] = len(inspect.signature(f).parameters)
            evaluate[l] = f
        else:
            signature[l] = 0
            evaluate[l] = lambda: f

    return EvaluableSignature(signature=signature, evaluate=evaluate, generate=generate)

def extend_evaluable_sig(sig, labels2funs):
    new_labels2funs = sig.evaluate.copy()
    new_labels2funs.update(labels2funs)
    return make_evaluable_sig(new_labels2funs, sig.generate)

def make_theory(sig, equalities):
    return Theory(sig, [parse(e) for e in equalities])

def assoc(label, identity=None):
    res = []
    res.append(make_equality(lambda x, y, z: (Term(label, [Term(label, [x, y]), z]),
                                              Term(label, [x, Term(label, [y, z])]))))
    if identity is not None:
        res.append(make_equality(lambda x: (Term(label, [Term(identity), x]), x)))
    return res

def comm(label, identity=None):
    res = []
    res.append(make_equality(lambda x, y: (Term(label, [x, y]),
                                           Term(label, [y, x]))))
    if identity is not None:
        res.append(make_equality(lambda x: (Term(label, [Term(identity), x]), x)))
    return res

def comm_assoc(label, identity=None):
    res = []
    res.append(make_equality(lambda x, y, z: (Term(label, [Term(label, [x, y]), z]),
                                              Term(label, [x, Term(label, [y, z])]))))
    res.append(make_equality(lambda x, y: (Term(label, [x, y]), Term(label, [y, x]))))
    if identity is not None:
        res.append(make_equality(lambda x: (Term(label, [Term(identity), x]), x)))
    return res

def distrib_left(mul_label, add_label):
    x, y, z = [Node() for i in range(3)]
    return [make_equality(lambda x, y, z: (Term(mul_label, [x, Term(add_label, [y, z])]),
                                           Term(add_label, [Term(mul_label, [x, y]),
                                                            Term(mul_label, [x, z])])))]

BooleanSig = \
    make_evaluable_sig(
        {'0': False,
         '1': True,
         'neg': lambda x: not x,
         'and': lambda x, y: x and y,
         'or' : lambda x, y: x or y},
        strategies.booleans)

BooleanExtSig = \
    extend_evaluable_sig(
        BooleanSig,
        {'impl': lambda x, y: (not x) or y,
         'xor': lambda x, y: x != y,
         'eq': lambda x, y: x == y})

BooleanTheory = \
    make_theory(
        BooleanSig.signature,
        ["neg(0) = 1", "neg(1) = 0"] +
        comm_assoc('or', 0) +
        comm_assoc('and', 1) +
        distrib_left('and', 'or') +
        distrib_left('or', 'and') +
        ["or(x, neg(x)) = 1", "and(x, neg(x)) = 0"] +
        ["or(x, x) = x", "and(x, x) = x"] +
        ["or(1, x) = 1", "and(0, x) = 0"] +
        ["or(x, and(x, y)) = x", "and(x, or(x, y)) = x"] +
        ["neg(or(x, y)) = and(neg(x), neg(y))", "neg(and(x, y)) = or(neg(x), neg(y))"] +
        ["neg(neg(x)) = x"])

