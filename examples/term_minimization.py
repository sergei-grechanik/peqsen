import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import pickle
import sys
import attr
import random
import os

@attr.s()
class MinimizedTerm:
    original : Term = attr.ib()
    minimized : Term = attr.ib()
    script : Script = attr.ib()

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

def boolean_terms(min_size=10, max_size=200, count_per_size=10, variables=3, seed=42):
    random.seed(seed)
    res = []
    signature = dict(BooleanSig.signature)
    for v in range(variables):
        signature[Var(v)] = 0
    for size in range(min_size, max_size):
        for _ in range(count_per_size):
            res.append(generate_term(signature, size))
    return res

def minimize_naively(term, script, theory, max_steps=1000):
    graph = Hypergraph()
    rewriter = Rewriter(graph)
    explanator = ExplanationTracker(graph)
    smallest_tracker = SmallestHyperedgeTracker(graph)

    for e in theory.equalities:
        rewriter.add_rule(EqualityRule(e))
        rewriter.add_rule(EqualityRule(e, reverse=True))

    added_elements = graph.rewrite(add=term)[1:]
    term_node = added_elements[0]

    top = Node()
    original_match = {k: v for k, v in zip(list_term_elements(term, top_node=top), added_elements)}

    if script is not None:
        res = run_script(graph, script)

    for i in range(max_steps):
        rewriter.perform_rewriting(max_n=1)

    smallest_term = list(smallest_tracker.smallest_terms(term_node.follow()))[0]

    smallest_match = list(smallest_tracker.smallest_matches(term_node.follow(), top_node=top))[0]
    smallest_match.update(original_match)
    script = explanator.script([smallest_match])

    return smallest_term, script

def minimize_repeatedly(term, theory, max_steps=1000):
    best_term = term
    best_script = None

    print(term)

    while True:
        cur_max_steps = \
            max_steps if best_script is None else max_steps - script_length(best_script)
        new_term, new_script = minimize_naively(term, script=best_script, theory=theory,
                                                max_steps=cur_max_steps)

        print(new_term)
        print("new term size", term_size(new_term),
              "script size", script_length(new_script))

        assert term_size(new_term) <= term_size(best_term)
        assert script_length(new_script) <= max_steps

        if best_script is None or term_size(new_term) < term_size(best_term) or \
                script_length(new_script) < script_length(best_script):
            best_term, best_script = new_term, new_script
            print("best term size", term_size(new_term),
                  "script size", script_length(new_script))
        else:
            break

        if script_length(new_script) >= max_steps:
            break

    return best_term, best_script
    print()

def some_naively_minimized_terms(val):
    tuples = []
    for t in list(boolean_terms(min_size=5, max_size=6)):
        newterm, script = minimize_repeatedly(t, BooleanTheory)
        tuples.append(MinimizedTerm(t, newterm, script))
    return tuples

def pickled(filename, func):
    filename = os.path.expanduser(os.path.join("~/.peqsen", filename))
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))
    else:
        res = func()
        pickle.dump(res, open(filename, 'wb'))
        return res

if __name__ == "__main__":
    for minimized in pickled("some_terms", lambda: some_naively_minimized_terms(10)):
        print()
        print(minimized.original)
        print(minimized.minimized)
        print(term_size(minimized.original), " -> ", term_size(minimized.minimized),
              "steps", script_length(minimized.script))

