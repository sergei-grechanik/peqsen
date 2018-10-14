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

if __name__ == "__main__":
    test_find_matches()
