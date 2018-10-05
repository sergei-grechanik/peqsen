from peqsen.hypergraph import Hypergraph, Node, Hyperedge, Term, Recursively
import peqsen.hypergraph as H
import hypothesis
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
    """Rewriting leaves graph in a consistent state. Also adding really adds and merging
    really merges, removing is tested separately
    (because addition has higher priority than removing)."""
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
    """Removing indeed removes"""
    h = data.draw(H.gen_hypergraph())
    rw = data.draw(H.gen_rewrite(h, max_add_hyperedges=0, max_merge=0))
    h.rewrite(**rw)
    hyperedges = h.hyperedges()
    for h in rw['remove']:
        assert not h in hyperedges

@given(strategies.data())
def test_add_from(data):
    """add_from results in isomorphic graph, and its mapping can be used to apply the
    same transformations to each copy. Moreover, the order of elements in the rewrite may be
    different."""
    h1 = data.draw(H.gen_hypergraph())
    h2 = Hypergraph()
    mapping = h2.add_from(h1)
    assert h1.isomorphic(h2)

    rw1 = data.draw(H.gen_rewrite(h1))
    rw2 = H.map_rewrite(data.draw(H.gen_permuted_rewrite(rw1)), mapping)

    h1.rewrite(**rw1)
    h2.rewrite(**rw2)

    assert h1.isomorphic(h2)

@given(strategies.data())
def test_add_removed(data):
    """Removing hyperedges and then adding the same hyperedges is noop."""
    h1 = data.draw(H.gen_hypergraph())
    h2 = Hypergraph()
    mapping = h2.add_from(h1)

    rw1 = data.draw(H.gen_rewrite(h1))
    rw2 = H.map_rewrite(rw1, mapping)
    rw2['remove'] = []

    h1.rewrite(**rw1)
    h1.rewrite(add=rw1['remove'])

    h2.rewrite(**rw2)

    assert h1.isomorphic(h2)

@given(strategies.data())
def test_rewrite_noremove_order(data):
    """Rewritings that don't remove may be applied in any order"""
    h1 = data.draw(H.gen_hypergraph())
    h2 = Hypergraph()
    h3 = Hypergraph()
    mapping12 = h2.add_from(h1)
    mapping23 = h3.add_from(h2)

    rwa = data.draw(H.gen_rewrite(h1, max_remove=0))
    rwb = data.draw(H.gen_rewrite(h1, max_remove=0))

    h1.rewrite(**rwa)
    h1.rewrite(**rwb)

    rwa2 = H.map_rewrite(rwa, mapping12)
    rwb2 = H.map_rewrite(rwb, mapping12)

    h2.rewrite(**rwb2)
    h2.rewrite(**rwa2)

    rwa3 = H.map_rewrite(rwa2, mapping23)
    rwb3 = H.map_rewrite(rwb2, mapping23)

    h3.rewrite(add=(rwa3['add'] + rwb3['add']), merge=(rwa3['merge'] + rwb3['merge']))

    assert h1.isomorphic(h2)
    assert h1.isomorphic(h3)

@given(strategies.data())
def test_rewrite_remove_order(data):
    """Rewritings that only remove hyperedges and add nodes may be applied in any order if
    we ignore already removed."""
    h1 = data.draw(H.gen_hypergraph())
    h2 = Hypergraph()
    h3 = Hypergraph()
    mapping12 = h2.add_from(h1)
    mapping23 = h3.add_from(h2)

    rwa = data.draw(H.gen_rewrite(h1, max_add_hyperedges=0, max_merge=0))
    rwb = data.draw(H.gen_rewrite(h1, max_add_hyperedges=0, max_merge=0))

    h1.rewrite(**rwa)
    h1.rewrite(**rwb, ignore_already_removed=True)

    rwa2 = H.map_rewrite(rwa, mapping12)
    rwb2 = H.map_rewrite(rwb, mapping12)

    h2.rewrite(**rwb2)
    h2.rewrite(**rwa2, ignore_already_removed=True)

    rwa3 = H.map_rewrite(rwa2, mapping23)
    rwb3 = H.map_rewrite(rwb2, mapping23)

    h3.rewrite(add=(rwa3['add'] + rwb3['add']), remove=(rwa3['remove'] + rwb3['remove']))

    assert h1.isomorphic(h2)
    assert h1.isomorphic(h3)

@given(strategies.data())
def test_listener(data):
    """This tests events. Note that the integrity is non-strict."""
    h1 = data.draw(H.gen_hypergraph())

    class _L:
        def __init__(self, to_add):
            self.to_add = to_add

        def on_add(self, hypergraph, elements):
            hypergraph.check_integrity(False)
            for e in elements:
                assert e in hypergraph
                assert e not in self.to_add

            self.to_add |= set(elements)

        def on_merge(self, hypergraph, node, removed, added):
            hypergraph.check_integrity(False)
            assert node not in hypergraph
            assert node.merged in hypergraph
            assert node in self.to_add
            assert node.merged in self.to_add
            for h in removed:
                assert h not in hypergraph
                assert h.merged in hypergraph
                assert h in self.to_add
            for h in added:
                assert h in hypergraph
                assert h not in self.to_add

            self.to_add -= set(removed)
            self.to_add -= set([node])
            self.to_add |= set(added)

        def on_remove(self, hypergraph, elements):
            hypergraph.check_integrity(False)
            for e in elements:
                assert e not in hypergraph
                assert e in self.to_add

            self.to_add -= set(elements)

    lis = _L(set(h1.nodes()) | set(h1.hyperedges()))
    h1.listeners.add(lis)

    rw = data.draw(H.gen_rewrite(h1))
    h1.rewrite(**rw)

    h2 = Hypergraph()
    h2.rewrite(add=lis.to_add)
    assert h1.isomorphic(h2)

@given(strategies.data())
def test_smallest_hyperedge_tracker(data):
    h1 = H.Hypergraph()
    tracker1 = H.BestHyperedgeTracker(measure=H.BestHyperedgeTracker.size)
    tracker2 = H.BestHyperedgeTracker(measure=H.BestHyperedgeTracker.depth)
    h1.listeners.add(tracker1)
    h1.listeners.add(tracker2)

    for i in range(2):
        rw = data.draw(H.gen_rewrite(h1))
        h1.rewrite(**rw)
        if data.draw(strategies.booleans()):
            h1.remove_nodes(data.draw(strategies.sampled_from(list(h1.nodes()))))

    h1.check_integrity()

    for n in h1.nodes():
        terms = [(H.measure_term(t, tracker1.measure), H.measure_term(t, tracker2.measure), t)
                 for t in H.finite_terms(n)]
        if terms:
            (min_val1, _, min_term1) = min(terms, key=lambda x: x[0])
            (_, min_val2, min_term2) = min(terms, key=lambda x: x[1])
            assert min_val1 == tracker1.best[n][0]
            assert min_val2 == tracker2.best[n][0]

            best1 = set(t for v, _, t in terms if v == min_val1)
            best2 = set(t for _, v, t in terms if v == min_val2)
            assert set(tracker1.best_terms(n)) == best1
            assert set(tracker2.best_terms(n)) == best2

            hypothesis.event("Number of smallest terms {}".format(len(best1)))
            hypothesis.event("Number of shallowest terms {}".format(len(best2)))
            hypothesis.event("Best by size == best by depth {}".format(best1 == best2))
        else:
            assert tracker1.best[n][0] == tracker1.worst_value
            assert tracker2.best[n][0] == tracker2.worst_value
            hypothesis.event("Number of smallest terms 0")
            hypothesis.event("Number of shallowest terms 0")
            hypothesis.event("Best by size == best by depth True")

if __name__ == "__main__":
    test_basic_operations()
    test_simple_addition()
    test_simple_addition_twice()
    test_add_ones_own()
    test_rewriting()
    test_remove()
    test_add_from()
    test_add_removed()
    test_rewrite_noremove_order()
    test_rewrite_remove_order()
    test_listener()
    test_smallest_hyperedge_tracker()
