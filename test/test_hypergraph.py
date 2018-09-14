from peqsen.hypergraph import Hypergraph, Node, Hyperedge, Term, Recursively
import peqsen.hypergraph as H
from hypothesis import given, strategies, reproduce_failure

def test_basic_operations():
    hg = Hypergraph()
    [ha] = hg.rewrite(add=Hyperedge('a', dst=[]))
    [hb1, hb2] = hg.rewrite(add=[Hyperedge('b', dst=[ha.src]),
                                 Hyperedge('b', dst=[ha.src, ha.src])])
    [hc1, hc2] = hg.rewrite(add=[Hyperedge('c', dst=[hb1.src]),
                                 Hyperedge('c', dst=[hb2.src])])
    hg.rewrite(merge=[(hb1.src, hb2.src)])
    hg.rewrite(remove=[hb1])
    hg_first = hg

    hg = Hypergraph()
    [ha] = hg.rewrite(add=Term('a'))
    [hb1, hb2] = hg.rewrite(add=[Term('b', [ha.src]),
                                 Term('b', [ha.src, ha.src])])
    [hc1, hc2] = hg.rewrite(add=[Term(('c', hb1.src)),
                                 Term(('c', hb2.src))])
    hg.rewrite(merge=[(hb1.src, hb2.src)])
    assert not hg_first.isomorphic(hg)
    hg.rewrite(remove=[hb1])

    assert hg_first.isomorphic(hg)

    hg3 = Hypergraph()
    hg3.rewrite(add=[Recursively(hc1.src)])

    assert hg_first.isomorphic(hg3)

@given(strategies.data())
def test_simple_addition(data):
    h1 = Hypergraph()
    h2 = Hypergraph()

    to_add1 = data.draw(H.simple_addition_strategy())
    to_add2 = data.draw(strategies.permutations(to_add1))

    h1.rewrite(add=to_add1)
    h2.rewrite(add=to_add2)

    iso = h1.isomorphic(h2)

    if not iso:
        print(h1)
        print(h2)

    assert iso

@given(strategies.data())
def test_simple_addition_twice(data):
    h1 = Hypergraph()
    h2 = Hypergraph()

    to_add1 = data.draw(H.simple_addition_strategy())
    to_add2 = data.draw(H.simple_addition_strategy())

    h1.rewrite(add=to_add1)
    h2.rewrite(add=data.draw(strategies.permutations(to_add1)))

    h1.check_integrity()

    h1.rewrite(add=to_add2)
    h2.rewrite(add=data.draw(strategies.permutations(to_add2)))

    h1.check_integrity()

    iso = h1.isomorphic(h2)

    if not iso:
        print(h1)
        print(h2)

    assert iso


if __name__ == "__main__":
    test_basic_operations()
    test_simple_addition()
    test_simple_addition_twice()
