import peqsen as PE
from peqsen import *
import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

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
    na, *_ = hg.rewrite(add=term('a', ()))
    nb1, nb2, *_ = hg.rewrite(add=[term('b', [na]),
                                   term('b', [na, na])])
    hb1 = list(nb1.outgoing)[0]
    nc1, nc2, *_ = hg.rewrite(add=[term(('c', nb1)),
                                   term(('c', nb2))])
    hg.rewrite(merge=[(nb1, nb2)])
    assert not hg_first.isomorphic(hg)
    hg.rewrite(remove=[hb1])

    assert hg_first.isomorphic(hg)

    hg3 = Hypergraph()
    hg3.rewrite(add=list_descendants(nc1.follow()))

    assert hg_first.isomorphic(hg3)

    n1 = Node()
    n2 = Node()
    n3 = Node()

    hg1 = Hypergraph()
    hg2 = Hypergraph()
    hg1.rewrite(add=[Hyperedge('a', src=n1, dst=[n2, n3]), Hyperedge('b', src=n2, dst=[])])
    hg2.rewrite(add=[Hyperedge('a', src=n1, dst=[n3, n2]), Hyperedge('b', src=n2, dst=[])])
    assert not hg1.isomorphic(hg2)

    hg1 = Hypergraph()
    hg2 = Hypergraph()
    hg1.rewrite(add=[Hyperedge('a', src=n1, dst=[n2, n2])])
    hg2.rewrite(add=[Hyperedge('a', src=n1, dst=[n2])])
    assert not hg1.isomorphic(hg2)

    hg1 = Hypergraph()
    hg2 = Hypergraph()
    hg1.rewrite(add=[Hyperedge('a', src=n1, dst=[n2, n3])])
    hg2.rewrite(add=[Hyperedge('b', src=n1, dst=[n2, n3])])
    assert not hg1.isomorphic(hg2)

@given(strategies.data())
def test_simple_addition(data):
    for congruence in [True, False]:
        h1 = Hypergraph(congruence=congruence)
        h2 = Hypergraph(congruence=congruence)

        to_add1 = data.draw(PE.gen_simple_addition())
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
    for congruence in [True, False]:
        h1 = Hypergraph(congruence=congruence)
        h2 = Hypergraph(congruence=congruence)

        to_add1 = data.draw(PE.gen_simple_addition())
        to_add2 = data.draw(PE.gen_simple_addition())

        h1.rewrite(add=to_add1)
        h2.rewrite(add=data.draw(strategies.permutations(to_add1)))

        h1.check_integrity()

        h1.rewrite(add=to_add2)
        h2.rewrite(add=data.draw(strategies.permutations(to_add2)))

        h1.check_integrity()

        assert h1.isomorphic(h2)

@given(strategies.data())
def test_add_ones_own(data):
    """Adding its own edges and nodes to hypergraph will not change it.
    This is true only for congruently closed graphs."""
    h1 = Hypergraph()
    h2 = Hypergraph()
    to_add = data.draw(PE.gen_simple_addition())
    h1.rewrite(add=to_add)
    h2.rewrite(add=to_add)
    assert h1.isomorphic(h2)

    nodes = list(h1.nodes())
    add_nodes = data.draw(strategies.lists(strategies.sampled_from(nodes),
                                           max_size=len(nodes)))
    recursive = data.draw(strategies.lists(strategies.booleans(),
                                           min_size=len(add_nodes), max_size=len(add_nodes)))
    add_nodes = [e for n, r in zip(add_nodes, recursive)
                 for e in (list_descendants(n) if r else [n])]

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
    for congruence in [True, False]:
        h = data.draw(PE.gen_hypergraph(congruence=congruence))
        for i in range(2):
            rw = data.draw(PE.gen_rewrite(h))
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
    for congruence in [True, False]:
        h = data.draw(PE.gen_hypergraph(congruence=congruence))
        rw = data.draw(PE.gen_rewrite(h, num_add_hyperedges=0, num_merge=0))
        h.rewrite(**rw)
        hyperedges = h.hyperedges()
        for h in rw['remove']:
            assert not h in hyperedges

@given(strategies.data())
def test_add_from(data):
    """add_from results in isomorphic graph, and its mapping can be used to apply the
    same transformations to each copy. Moreover, the order of elements in the rewrite may be
    different."""
    for congruence in [True, False]:
        h1 = data.draw(PE.gen_hypergraph(congruence=congruence))
        h2 = Hypergraph(congruence=congruence)
        mapping = h2.add_from(h1)
        assert h1.isomorphic(h2)

        rw1 = data.draw(PE.gen_rewrite(h1))
        rw2 = PE.map_rewrite(data.draw(PE.gen_permuted_rewrite(rw1)), mapping)

        h1.rewrite(**rw1)
        h2.rewrite(**rw2)

        assert h1.isomorphic(h2)

@given(strategies.data())
def test_add_removed(data):
    """Removing hyperedges and then adding the same hyperedges is noop."""
    for congruence in [True, False]:
        h1 = data.draw(PE.gen_hypergraph(congruence=congruence))
        h2 = Hypergraph(congruence=congruence)
        mapping = h2.add_from(h1)

        rw1 = data.draw(PE.gen_rewrite(h1))
        rw2 = PE.map_rewrite(rw1, mapping)
        rw2['remove'] = []

        h1.rewrite(**rw1)
        if congruence:
            h1.rewrite(add=rw1['remove'])
        else:
            # For the noncongruent case we have to have to be careful with duplicates
            h1.rewrite(add=set(rw1['remove']))

        h2.rewrite(**rw2)

        assert h1.isomorphic(h2)

@given(strategies.data())
def test_rewrite_noremove_order(data):
    """Rewritings that don't remove may be applied in any order"""
    for congruence in [True, False]:
        h1 = data.draw(PE.gen_hypergraph(congruence=congruence))
        h2 = Hypergraph(congruence=congruence)
        h3 = Hypergraph(congruence=congruence)
        mapping12 = h2.add_from(h1)
        mapping23 = h3.add_from(h2)

        rwa = data.draw(PE.gen_rewrite(h1, num_remove=0))
        rwb = data.draw(PE.gen_rewrite(h1, num_remove=0))

        h1.rewrite(**rwa)
        h1.rewrite(**rwb)

        rwa2 = PE.map_rewrite(rwa, mapping12)
        rwb2 = PE.map_rewrite(rwb, mapping12)

        h2.rewrite(**rwb2)
        h2.rewrite(**rwa2)

        rwa3 = PE.map_rewrite(rwa2, mapping23)
        rwb3 = PE.map_rewrite(rwb2, mapping23)

        h3.rewrite(add=(rwa3['add'] + rwb3['add']), merge=(rwa3['merge'] + rwb3['merge']))

        assert h1.isomorphic(h2)
        assert h1.isomorphic(h3)

@given(strategies.data())
def test_rewrite_remove_order(data):
    """Rewritings that only remove hyperedges and add nodes may be applied in any order if
    we ignore already removed."""
    for congruence in [True, False]:
        h1 = data.draw(PE.gen_hypergraph(congruence=congruence))
        h2 = Hypergraph(congruence=congruence)
        h3 = Hypergraph(congruence=congruence)
        mapping12 = h2.add_from(h1)
        mapping23 = h3.add_from(h2)

        rwa = data.draw(PE.gen_rewrite(h1, num_add_hyperedges=0, num_merge=0))
        rwb = data.draw(PE.gen_rewrite(h1, num_add_hyperedges=0, num_merge=0))

        h1.rewrite(**rwa)
        h1.rewrite(**rwb, ignore_already_removed=True)

        rwa2 = PE.map_rewrite(rwa, mapping12)
        rwb2 = PE.map_rewrite(rwb, mapping12)

        h2.rewrite(**rwb2)
        h2.rewrite(**rwa2, ignore_already_removed=True)

        rwa3 = PE.map_rewrite(rwa2, mapping23)
        rwb3 = PE.map_rewrite(rwb2, mapping23)

        h3.rewrite(add=(rwa3['add'] + rwb3['add']), remove=(rwa3['remove'] + rwb3['remove']))

        assert h1.isomorphic(h2)
        assert h1.isomorphic(h3)

@given(strategies.data())
def test_listener(data):
    """This tests events. Note that the integrity is non-strict."""
    for congruence in [True, False]:
        h1 = data.draw(PE.gen_hypergraph(congruence=congruence))

        class _L:
            def __init__(self, to_add):
                self.to_add = to_add

            def on_add(self, hypergraph, elements):
                hypergraph.check_integrity(False)
                for e in elements:
                    assert e in hypergraph
                    assert e not in self.to_add

                self.to_add |= set(elements)

            def on_merge(self, hypergraph, node, removed, added, reason):
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

        rw = data.draw(PE.gen_rewrite(h1))
        h1.rewrite(**rw)

        h2 = Hypergraph(congruence=congruence)
        h2.rewrite(add=lis.to_add)
        assert h1.isomorphic(h2)

@given(strategies.data())
def test_smallest_hyperedge_tracker(data):
    for congruence in [True, False]:
        PE.GloballyIndexed.reset_global_index()
        h1 = PE.Hypergraph(congruence=congruence)
        tracker1 = PE.SmallestHyperedgeTracker(measure=PE.SmallestHyperedgeTracker.size)
        tracker2 = PE.SmallestHyperedgeTracker(measure=PE.SmallestHyperedgeTracker.depth)
        h1.listeners.add(tracker1)
        h1.listeners.add(tracker2)

        max_number_of_smallest = 0
        there_was_by_size_ineq_by_depth = False

        for i in range(data.draw(strategies.integers(2, 5))):
            rw = data.draw(PE.gen_rewrite(h1))
            h1.rewrite(**rw)
            if data.draw(strategies.booleans()):
                h1.remove_nodes(data.draw(strategies.sampled_from(list(h1.nodes()))))

            for n in h1.nodes():
                # TODO: Sometimes there are just too many terms. In this case we assume false
                # but this isn't very elegant. A better approach is to find smallest terms instead
                # of enumerating all terms.
                terms = []
                for t in PE.finite_terms(n):
                    i = i + 1
                    if i > 1000:
                        print("Ooups, too many terms")
                        hypothesis.assume(False)
                    terms.append((PE.measure_term(t, tracker1.measure),
                                  PE.measure_term(t, tracker2.measure),
                                  t))

                if terms:
                    (min_val1, _, min_term1) = min(terms, key=lambda x: x[0])
                    (_, min_val2, min_term2) = min(terms, key=lambda x: x[1])

                    assert min_val1 == tracker1.smallest[n][0]
                    assert min_val2 == tracker2.smallest[n][0]

                    smallest1 = set(t for v, _, t in terms if v == min_val1)
                    smallest2 = set(t for _, v, t in terms if v == min_val2)
                    assert set(tracker1.smallest_terms(n)) == smallest1
                    # For depth tracker will not return the full set of shallowest terms
                    assert set(tracker2.smallest_terms(n)).issubset(smallest2)

                    max_number_of_smallest = max(max_number_of_smallest, len(smallest1))
                    max_number_of_smallest = max(max_number_of_smallest, len(smallest2))
                    if smallest1 != smallest2:
                        there_was_by_size_ineq_by_depth = True
                else:
                    assert tracker1.smallest[n][0] == tracker1.worst_value
                    assert tracker2.smallest[n][0] == tracker2.worst_value

        hypothesis.event("Max number of smallest: " + str(max_number_of_smallest))
        hypothesis.event("There was a node where the num of smallest by size != by depth: " +
                         str(there_was_by_size_ineq_by_depth))

@given(strategies.data())
def test_pattern_is_acyclic(data):
    n, h = data.draw(PE.gen_pattern())

    assert n in h

    checked = set()

    def _check(node, history):
        if node not in checked:
            if node in history:
                raise AssertionError("Recursive node detected: {}".format(node))
            history = history + [node]
            for h in node.outgoing:
                for n in h.dst:
                    _check(n, history)
            checked.add(node)

    for n in h.nodes():
        _check(n, [])


@given(strategies.data())
def test_stat(data):
    """This isn't really a test. Here we simply show the stats of creating a hypergraph and
    transforming it"""
    PE.GloballyIndexed.reset_global_index()

    h = Hypergraph()

    nodes_stat = []
    hyperedges_stat = []

    for i in range(5):
        rw = data.draw(PE.gen_rewrite(h))
        h.rewrite(**rw)
        nodes_stat.append(len(h.nodes()))
        hyperedges_stat.append(len(h.hyperedges()))

    def _str(val):
        if val >= 5:
            if val > 10:
                return "> 10"
            else:
                return "5-10"
        else:
            return str(val)

    hypothesis.event("Max nodes: " + _str(max(nodes_stat)))
    hypothesis.event("Max hyperedges: " + _str(max(hyperedges_stat)))

    # hypothesis.event("Final nodes: " + _str(len(h.nodes())))
    # hypothesis.event("Final hyperedges: " + _str(len(h.hyperedges())))

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
    test_pattern_is_acyclic()
    test_stat()
