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

    to_add1 = data.draw(H.gen_simple_addition())
    to_add2 = data.draw(strategies.permutations(to_add1))

    added1 = h1.rewrite(add=to_add1)
    h2.rewrite(add=to_add2)

    assert len(added1) == len(to_add1)

    for e in added1:
        if isinstance(e, Node):
            assert e in h1.nodes()
        else:
            assert e in h1.hyperedges()

    assert h1.isomorphic(h2)

@given(strategies.data())
def test_simple_addition_twice(data):
    h1 = Hypergraph()
    h2 = Hypergraph()

    to_add1 = data.draw(H.gen_simple_addition())
    to_add2 = data.draw(H.gen_simple_addition())

    h1.rewrite(add=to_add1)
    h2.rewrite(add=data.draw(strategies.permutations(to_add1)))

    h1.check_integrity()

    h1.rewrite(add=to_add2)
    h2.rewrite(add=data.draw(strategies.permutations(to_add2)))

    h1.check_integrity()

    assert h1.isomorphic(h2)

@given(strategies.data())
def test_add_ones_own(data):
    """Adding its own edges and nodes to hypergraph will not change it"""
    h1 = Hypergraph()
    h2 = Hypergraph()
    to_add = data.draw(H.gen_simple_addition())
    h1.rewrite(add=to_add)
    h2.rewrite(add=to_add)
    assert h1.isomorphic(h2)

    nodes = list(h1.nodes())
    add_nodes = data.draw(strategies.lists(strategies.sampled_from(nodes),
                                           max_size=len(nodes)))
    recursive = data.draw(strategies.lists(strategies.booleans(),
                                           min_size=len(add_nodes), max_size=len(add_nodes)))
    add_nodes = [Recursively(n) if r else n for n, r in zip(add_nodes, recursive)]

    hyperedges = list(h1.hyperedges())
    add_hyperedges = data.draw(strategies.lists(strategies.sampled_from(hyperedges),
                                                max_size=len(hyperedges)))

    h1.rewrite(add=add_nodes + add_hyperedges)
    h1.check_integrity()
    assert h1.isomorphic(h2)

    # This won't work for non-terms
    #h2.rewrite(add=add_nodes + add_hyperedges)
    #h2.check_integrity()
    #assert h1.isomorphic(h2)

@given(strategies.data())
def test_rewriting(data):
    h = data.draw(H.gen_hypergraph())
    for i in range(2):
        rw = data.draw(H.gen_rewrite(h))
        added = h.rewrite(**rw)
        h.check_integrity()

        nodes = h.nodes()
        hyperedges = h.hyperedges()

        for n1, n2 in rw['merge']:
            n1 = n1.follow()
            n2 = n2.follow()
            assert n1 == n2
            assert n1 in nodes

        for e in added:
            if isinstance(e, Node):
                assert e in nodes
            else:
                assert e in hyperedges

@given(strategies.data())
def test_remove(data):
    h = data.draw(H.gen_hypergraph())
    rw = data.draw(H.gen_rewrite(h, max_add_hyperedges=0, max_merge=0))
    h.rewrite(**rw)
    hyperedges = h.hyperedges()
    for h in rw['remove']:
        assert not h in hyperedges

if __name__ == "__main__":
    test_basic_operations()
    test_simple_addition()
    test_simple_addition_twice()
    test_add_ones_own()
    test_rewriting()
    test_remove()
