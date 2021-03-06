
import hypothesis
from hypothesis import strategies

import peqsen.util
from peqsen import Node, Hyperedge, Hypergraph, term, gen_list, parse, make_equality

import inspect
import attr

@attr.s(frozen=True)
class FiniteFunction:
    name = attr.ib()
    arity = attr.ib()
    carrier_size = attr.ib()
    table = attr.ib()
    func = attr.ib()

@attr.s(frozen=True)
class EvaluableSignature:
    signature = attr.ib()
    evaluate = attr.ib()
    generate = attr.ib()

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

@attr.s(slots=True, frozen=True)
class Theory:
    signature = attr.ib()
    equalities = attr.ib()

def make_evaluable_sig(labels2funs, generate, propagate_none=False):
    signature = {}
    evaluate = {}
    for l, f in labels2funs.items():
        if callable(f):
            signature[l] = len(inspect.signature(f).parameters)
            if propagate_none:
                f = lambda *args, f=f: None if None in args else f(*args)
            evaluate[l] = f
        else:
            signature[l] = 0
            evaluate[l] = lambda f=f: f

    return EvaluableSignature(signature=signature, evaluate=evaluate, generate=generate)

def extend_evaluable_sig(sig, labels2funs):
    new_labels2funs = sig.evaluate.copy()
    new_labels2funs.update(labels2funs)
    return make_evaluable_sig(new_labels2funs, sig.generate)

def make_theory(sig, equalities):
    return Theory(sig, [parse(e) for e in equalities])

def assoc(label, identity=None):
    res = []
    res.append(make_equality(lambda x, y, z: (term(label, [term(label, [x, y]), z]),
                                              term(label, [x, term(label, [y, z])]))))
    if identity is not None:
        res.append(make_equality(lambda x: (term(label, [term(identity, ()), x]), x)))
        res.append(make_equality(lambda x: (term(label, [x, term(identity, ())]), x)))
    return res

def comm(label, identity=None):
    res = []
    res.append(make_equality(lambda x, y: (term(label, [x, y]),
                                           term(label, [y, x]))))
    if identity is not None:
        res.append(make_equality(lambda x: (term(label, [term(identity), x]), x)))
    return res

def comm_assoc(label, identity=None):
    res = []
    res.append(make_equality(lambda x, y, z: (term(label, [term(label, [x, y]), z]),
                                              term(label, [x, term(label, [y, z])]))))
    res.append(make_equality(lambda x, y: (term(label, [x, y]), term(label, [y, x]))))
    if identity is not None:
        res.append(make_equality(lambda x: (term(label, [term(identity, ()), x]), x)))
    return res

def distrib_left(mul_label, add_label):
    x, y, z = [Node() for i in range(3)]
    return [make_equality(lambda x, y, z: (term(mul_label, [x, term(add_label, [y, z])]),
                                           term(add_label, [term(mul_label, [x, y]),
                                                            term(mul_label, [x, z])])))]

SKISig = {'S': 0, 'K': 0, 'I': 0, 'app': 2}

SKITheory = \
    make_theory(
        SKISig,
        ["app(I, x) = x",
         "app(app(K, x), y) = y",
         "app(app(app(S, x), y), z) = app(app(x, y), app(x, z))"])

GroupSig = {'1': 0, 'mul': 2, 'inv': 1}

GroupTheory = \
    make_theory(
        GroupSig,
        assoc('mul', '1') +
        ["mul(inv(x), x) = 1", "mul(x, inv(x)) = 1"] +
        ["inv(mul(x, y)) = mul(inv(y), inv(x))"])

AbelianGroupTheory = \
    make_theory(
        GroupSig,
        comm_assoc('mul', '1') +
        ["mul(inv(x), x) = 1"] +
        ["inv(mul(x, y)) = mul(inv(y), inv(x))"])

BooleanSig = \
    make_evaluable_sig(
        {'0': False,
         '1': True,
         'not': lambda x: not x,
         'and': lambda x, y: x and y,
         'or' : lambda x, y: x or y},
        strategies.booleans())

BooleanExtSig = \
    extend_evaluable_sig(
        BooleanSig,
        {'impl': lambda x, y: (not x) or y,
         'xor': lambda x, y: x != y,
         'eq': lambda x, y: x == y})

BooleanTheory = \
    make_theory(
        BooleanSig.signature,
        ["not(0) = 1", "not(1) = 0"] +
        comm_assoc('or', '0') +
        comm_assoc('and', '1') +
        distrib_left('and', 'or') +
        distrib_left('or', 'and') +
        ["or(x, not(x)) = 1", "and(x, not(x)) = 0"] +
        ["or(x, x) = x", "and(x, x) = x"] +
        ["or(1, x) = 1", "and(0, x) = 0"] +
        ["or(x, and(x, y)) = x", "and(x, or(x, y)) = x"] +
        ["not(or(x, y)) = and(not(x), not(y))", "not(and(x, y)) = or(not(x), not(y))"] +
        ["not(not(x)) = x"])

BooleanExtTheory = \
    make_theory(
        BooleanExtSig.signature,
        BooleanTheory.equalities +
        ["impl(x, y) = or(not(x), y)",
         "impl(x, impl(y, z)) = impl(impl(x, y), impl(x, z))"] +
        comm_assoc('xor', '0') +
        ["xor(1, x) = not(x)"] +
        ["eq(x, y) = not(xor(x, y))"])

IntegerSig = \
    make_evaluable_sig(
        {'0': 0,
         '1': 1,
         'add': lambda x, y: x + y,
         'mul': lambda x, y: x * y,
         'neg': lambda x: -x},
        strategies.integers())

IntegerTheory = \
    make_theory(
        IntegerSig.signature,
        comm_assoc('add', '0') +
        comm_assoc('mul', '1') +
        distrib_left('mul', 'add') +
        ["add(x, neg(x)) = 0",
         "neg(neg(x)) = x",
         "mul(neg(x), y) = neg(mul(x, y))"])
