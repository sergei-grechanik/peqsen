
import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import sys

def test_parse():
    print(parse('f(x, g(y), lalal(), 12_1(z), A, 0) = g(y, x, x)'))
    print(parse('f(serer3, g(perer2,), lalal(), 12_1(terer1,), A(), 0()) = g(perer2, serer3, serer3)'))

if __name__ == "__main__":
    test_parse()
