from peqsen.hypergraph import Hypergraph, Node, Hyperedge, Term, Recursively
from hypothesis import given

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



if __name__ == "__main__":
    test_basic_operations()
