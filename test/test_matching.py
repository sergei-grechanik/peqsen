import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

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

@reproduce_failure('3.73.3', b'AXicY2ZkYGZkZGBgZOACkkCKmQETMLICCQADkQAg')
@given(strategies.data())
def test_trigger_manager_nondestructive(data):
    GloballyIndexed.reset_global_index()

    h = Hypergraph()
    p, hp = data.draw(gen_pattern())
    trigman = TriggerManager(h)
    trig_matches = []
    trigman.add_trigger(p, lambda m: trig_matches.append(m))

    for i in range(3):
        rw = data.draw(gen_rewrite(h, max_remove=0))
        print()
        print("Curent hypergraph")
        print(h)
        print(rw)
        print()
        h.rewrite(**rw)
        matches = list(find_matches(h, p))
        trig_matches = [match_follow(m) for m in trig_matches]
        print()
        print(h)
        print("matches", matches)
        print("trig_matches", trig_matches)
        print()
        assert list_of_matches_equal(matches, trig_matches)

if __name__ == "__main__":
    # test_find_matches()
    # test_pattern_self_match()
    test_trigger_manager_nondestructive()
