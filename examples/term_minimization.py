import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import sys
import attr

import joblib
import random
import os

def generate_term(signature, size, variables=None):
    if variables is None:
        variables = []
    elif isinstance(variables, int):
        variables = [Node() for _ in range(variables)]

    if size <= 1:
        leaves = variables + [Term(l) for l, a in signature.items() if a == 0]
        return random.choice(leaves)
    else:
        label, arity = random.choice([(l, a) for l, a in signature.items()
                                      if a + 1 <= size and a > 0])
        buckets = [1 for _ in range(arity)]
        for _ in range(size - arity - 1):
            buckets[random.randrange(arity)] += 1
        subterms = [generate_term(signature, subsize, variables) for subsize in buckets]
        return Term(label, subterms)

memory = joblib.Memory("~/.peqsen/")

@memory.cache
def boolean_terms(min_size=10, max_size=200, count_per_size=10, variables=3, seed=42):
    random.seed(seed)
    res = []
    for size in range(min_size, max_size):
        for _ in range(count_per_size):
            res.append(generate_term(BooleanSig.signature, size, variables))
    return res


if __name__ == "__main__":
    for t in boolean_terms():
        print(t)

