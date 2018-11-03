import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import sys

@given(strategies.data())
def test_find_matches(data):
    h = data.draw(gen_hypergraph())
    p, hp = data.draw(gen_pattern())
    assert isinstance(p, Node)
    assert len(p.incoming) == 0
    matches = list(find_matches(h, p))
    hypothesis.event("Number of matches {}".format(len(matches)))
    for m in matches:
        for pe, he in m.items():
            assert pe in hp
            assert he in h

    # adding and merging doesn't lead to matches' disappearing
    rw = data.draw(gen_rewrite(h, max_remove=0))
    h.rewrite(**rw)
    matches2 = list(find_matches(h, p))

    # if we then follow the elements, we get one of the new matches
    for m in matches:
        m_new = {k: m[k].follow() for k in m}
        assert m_new in matches2

@given(strategies.data())
def test_pattern_self_match(data):
    h = Hypergraph()
    p, hp = data.draw(gen_pattern())
    # Note that h may differ from the pattern graph, not only because of redundant nodes in hp,
    # but also because of congruence closure
    h.rewrite(add=Recursively(p))
    matches = list(find_matches(h, p))
    assert len(matches) > 0

    # adding and merging doesn't lead to matches' disappearing
    rw = data.draw(gen_rewrite(h, max_remove=0))
    h.rewrite(**rw)
    matches2 = list(find_matches(h, p))
    assert len(matches2) > 0

    hypothesis.event("Number of matches after rw {}".format(len(matches2)))

    # if we then follow the elements, we get one of the new matches
    for m in matches:
        m_new = {k: m[k].follow() for k in m}
        assert m_new in matches2

def list_of_matches_equal(matches1, matches2):
    for m in matches1:
        if m not in matches2:
            return False
    for m in matches2:
        if m not in matches1:
            return False
    return True

@given(strategies.data())
def test_trigger_manager_nondestructive(data):
    GloballyIndexed.reset_global_index()

    h = Hypergraph()
    p, hp = data.draw(gen_pattern())
    trigman = TriggerManager(h)
    trig_matches = []
    trigman.add_trigger(p, lambda m: trig_matches.append(m))

    for i in range(5):
        rw = data.draw(gen_rewrite(h, max_remove=0))
        h.rewrite(**rw)
        matches = list(find_matches(h, p))
        trig_matches = [match_follow(m) for m in trig_matches]
        assert list_of_matches_equal(matches, trig_matches)

    hypothesis.event("Final number of matches {}".format(len(matches)))

@given(strategies.data())
def test_trigger_manager_nondestructive_multipattern(data):
    GloballyIndexed.reset_global_index()

    h = Hypergraph()
    plist = data.draw(strategies.lists(gen_pattern(), min_size=1, max_size=4))
    trigman = TriggerManager(h)
    trig_matches = [[] for _ in plist]
    for i, (p, hp) in enumerate(plist):
        trigman.add_trigger(p, lambda m, i=i: trig_matches[i].append(m))

    for i in range(4):
        rw = data.draw(gen_rewrite(h, max_remove=0))
        h.rewrite(**rw)
        for trigm, (p, _) in zip(trig_matches, plist):
            matches = list(find_matches(h, p))
            trigm = [match_follow(m) for m in trigm]
            assert list_of_matches_equal(matches, trigm)

    hypothesis.event("Final number of matches {}".format(sum(len(l) for l in trig_matches)))

@given(strategies.data())
def test_trigger_manager_destructive(data):
    GloballyIndexed.reset_global_index()

    h = Hypergraph()
    plist = data.draw(strategies.lists(gen_pattern(), min_size=1, max_size=4))
    trigman = TriggerManager(h)
    trig_matches = [[] for _ in plist]
    for i, (p, hp) in enumerate(plist):
        trigman.add_trigger(p, lambda m, i=i: trig_matches[i].append(m))

    for i in range(4):
        rw = data.draw(gen_rewrite(h))
        # print()
        # print("Curent hypergraph")
        # print(h)
        # print(rw)
        # print()
        h.rewrite(**rw)
        for trigm, (p, _) in zip(trig_matches, plist):
            matches = list(find_matches(h, p))
            trigm = [match_follow(m) for m in trigm]
            # print()
            # print(h)
            # print("matches", matches)
            # print("trig_matches", trigm)
            # print()
            assert list_of_matches_equal(matches, trigm)
            trigm.clear()


@given(strategies.data())
def test_pattern_stats(data):
    h = data.draw(gen_hypergraph())
    p, hp = data.draw(gen_pattern())
    hypothesis.event("Outgoing: " + str(len(p.outgoing)))

    def _hyperedges(pat):
        if isinstance(pat, Node):
            return sum(_hyperedges(h) for h in pat.outgoing)
        else:
            return 1 + sum(_hyperedges(n) for n in pat.dst)

    hypothesis.event("Hyperedges: " + str(_hyperedges(p)))

if __name__ == "__main__":
    test_find_matches()
    test_pattern_self_match()
    test_trigger_manager_nondestructive()
    test_trigger_manager_nondestructive_multipattern()
    # test_trigger_manager_destructive()
    test_pattern_stats()
