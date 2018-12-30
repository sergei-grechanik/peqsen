
import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import sys

@given(strategies.data())
def test_parse(data):
    t1 = data.draw(gen_term())
    t2 = parse(str(t1))
    h1 = Hypergraph()
    h2 = Hypergraph()
    h1.rewrite(add=t1)
    h2.rewrite(add=t2)
    assert h1.isomorphic(h2)

@given(strategies.data())
def test_parse_eq(data):
    (eq1_lhs, eq1_rhs) = data.draw(gen_term(equality=True))
    eq2 = parse(str(eq1_lhs) + " = " + str(eq1_rhs))
    h1 = Hypergraph()
    h2 = Hypergraph()
    [na1, nb1] = h1.rewrite(add=[eq1_lhs, eq1_rhs])
    h1.rewrite(merge=[(na1, nb1)])
    [na2, nb2] = h2.rewrite(add=[eq2.lhs, eq2.rhs])
    h2.rewrite(merge=[(na2, nb2)])
    assert h1.isomorphic(h2)

def check_theory(data, theory, evaluablesig=None):
    graph = Hypergraph()
    rewriter = Rewriter(graph)
    num_models = data.draw(strategies.integers(1, 20))
    evaluator = SimpleEvaluator(graph, evaluablesig, data=data, num_models=num_models)

    for e in theory.equalities:
        destr = data.draw(strategies.booleans())
        if data.draw(strategies.booleans()):
            rewriter.add_rule(equality_to_rule(e, destructive=destr))
        if data.draw(strategies.booleans()):
            rewriter.add_rule(equality_to_rule(e, reverse=True))

    vars_number = data.draw(strategies.integers(0, 4))
    variables = [Node() for _ in range(vars_number)]
    terms = data.draw(strategies.lists(gen_term(theory.signature, variables=variables), max_size=5))
    graph.rewrite(add=terms)

    for i in range(data.draw(strategies.integers(1, 20))):
        rewriter.perform_rewriting(max_n=10)

@given(strategies.data())
def test_boolean(data):
    check_theory(data, BooleanTheory, BooleanSig)

@given(strategies.data())
def test_boolean_ext(data):
    check_theory(data, BooleanExtTheory, BooleanExtSig)

if __name__ == "__main__":
    test_parse()
    test_parse_eq()
    test_boolean()
    test_boolean_ext()
