
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

def check_theory(draw, theory, evaluablesig):
    pass

@given(strategies.data())
def test_boolean(data):
    pass

if __name__ == "__main__":
    test_parse()
    test_parse_eq()
    test_boolean()
