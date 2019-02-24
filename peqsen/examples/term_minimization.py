import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import pickle
import sys
import attr
import random
import os
import sqlitedict
import multiprocessing
import shutil
import tqdm

from typing import Optional, Tuple, Union, Dict, Any, Iterable

@attr.s()
class MinimizedTerm:
    original: Term = attr.ib()
    minimized: Term = attr.ib()
    script: Optional[Script] = attr.ib(default=None)
    method: Optional['MinimizationMethod'] = attr.ib(default=None)
    base: Optional['MinimizedTerm'] = attr.ib(default=None)

    def __lt__(self, other):
        if term_size(self.minimized) < term_size(other):
            return True
        if term_size(self.minimized) == term_size(other):
            if self.script is not None:
                if other.script is None:
                    return True
                if script_length(self.script) < script_length(other.script):
                    return True
        return False

class MinimizationMethod:
    def minimize(self, term: Term, script: Optional[Script]=None) -> Tuple[Term, Script]:
        pass

class MinimizedTermDB:
    def __init__(self, filename):
        self.newly_created = not os.path.exists(filename)
        self._db = sqlitedict.SqliteDict(os.path.expanduser(filename), "minimized_terms")

    def __contains__(self, term):
        return str(term) in self._db

    def __getitem__(self, term):
        return self._db[str(term)]

    def __setitem__(self, term, value):
        self._db[str(term)] = value
        self._db.commit()

    def get(self, term, default=None):
        try:
            return self[term]
        except KeyError:
            return default

    def values(self):
        return self._db.values()

    def empty(self):
        return len(self._db) == 0

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

def _generate_term(arg):
    signature, size, seed = arg
    random.seed(seed)
    return generate_term(signature, size)

def generate_random_terms_db(db: MinimizedTermDB, signature: Dict[Any, int], *, n_jobs=4,
                             min_size=10, max_size=200, count_per_size=10, variables=3, seed=42):
    random.seed(seed)

    signature = dict(signature)
    for v in range(variables):
        signature[Var(v)] = 0

    pool = multiprocessing.Pool(n_jobs)
    args = [(signature, i, random.randint(0, sys.maxsize))
            for i in range(min_size, max_size) for _ in range(count_per_size)]
    desc = "Generating random terms size in [{}, {}]".format(min_size, max_size)
    for t in tqdm.tqdm(pool.imap_unordered(_generate_term, args), desc=desc, total=len(args)):
        db[t] = db.get(t, MinimizedTerm(t, t))

def _minimize_term(arg):
    minimizer, mterm, seed = arg
    random.seed(seed)
    try:
        new_term, script = minimizer.minimize(mterm.original, mterm.script)
        new_mterm = MinimizedTerm(mterm.original, new_term, script=script, base=mterm, method=minimizer)
        return new_mterm
    except Exception as e:
        return (e, arg)

def minimize_terms_db(db: MinimizedTermDB,
                      minimizer: MinimizationMethod,
                      minimized_terms: Iterable[MinimizedTerm],
                      *, n_jobs=4, seed=42):
    random.seed(seed)

    pool = multiprocessing.Pool(n_jobs)
    args = [(minimizer, mterm, random.randint(0, sys.maxsize))
            for mterm in minimized_terms]
    desc = "Minimizing terms with {}".format(type(minimizer).__name__)
    for mt in tqdm.tqdm(pool.imap_unordered(_minimize_term, args), desc=desc, total=len(args)):
        if isinstance(mt, tuple) and isinstance(mt[0], Exception):
            ex, (_, mterm, _) = mt
            print()
            print("Exception happened during term minimization:")
            print("term", mterm.original)
            print(mterm)
            print(ex)
            print()
        else:
            previous = db.get(mt.original)
            if previous is not None:
                db[mt.original] = min(previous, mt)
            else:
                db[mt.original] = mt

@attr.s()
class NaiveMinimizationMethod:
    theory = attr.ib()
    max_steps: int = attr.ib(default=1000)

    def minimize(self, term: Term, script: Optional[Script]=None) -> Tuple[Term, Script]:
        graph = Hypergraph()
        rewriter = Rewriter(graph)
        explanator = ExplanationTracker(graph)
        smallest_tracker = SmallestHyperedgeTracker(graph)

        for e in self.theory.equalities:
            rewriter.add_rule(EqualityRule(e))
            rewriter.add_rule(EqualityRule(e, reverse=True))

        added_elements = graph.rewrite(add=term)[1:]
        term_node = added_elements[0]

        top = Node()
        original_match = \
            {k: v for k, v in zip(list_term_elements(term, top_node=top), added_elements)}

        steps = self.max_steps

        if script is not None:
            res = run_script(graph, script)
            steps -= script_length(script)

        for i in range(steps):
            rewriter.perform_rewriting(max_n=1)

        smallest_term = list(smallest_tracker.smallest_terms(term_node.follow()))[0]

        smallest_match = list(smallest_tracker.smallest_matches(term_node.follow(), top_node=top))[0]
        smallest_match.update(original_match)
        script = explanator.script([smallest_match])

        return smallest_term, script

@attr.s()
class RepeatedMinimizationMethod:
    method: MinimizationMethod = attr.ib()
    max_steps: int = attr.ib(default=10)

    def minimize(self, term: Term, script: Optional[Script]=None) -> Tuple[Term, Script]:
        best_term = term
        best_script = script

        for i in range(self.max_steps):
            new_term, new_script = self.method.minimize(term, best_script)

            assert term_size(new_term) <= term_size(best_term)

            if best_script is None or term_size(new_term) < term_size(best_term) or \
                    script_length(new_script) < script_length(best_script):
                best_term, best_script = new_term, new_script
            else:
                break

        return best_term, best_script

def some_naively_minimized_terms(val):
    tuples = []
    for t in list(boolean_terms(min_size=5, max_size=6)):
        newterm, script = minimize_repeatedly(t, BooleanTheory)
        tuples.append(MinimizedTerm(t, newterm, script))
    return tuples

class BooleanTermMinimizationExperiment:
    def __init__(self, name, n_jobs=4):
        self.name = name
        self.dirname = os.path.expanduser("~/.peqsen/" + name)

        self.n_jobs = n_jobs

        try:
            os.makedirs(self.dirname)
        except OSError:
            pass

    def terms_db(self):
        db = MinimizedTermDB(self.dirname + "/terms")
        if db.empty():
            generate_random_terms_db(db, BooleanSig.signature, n_jobs=self.n_jobs)
        return db

    def naively_minimized_terms(self, run=True):
        db = MinimizedTermDB(self.dirname + "/naively-minimized")
        if run:
            terms_db = self.terms_db()
            method = RepeatedMinimizationMethod(NaiveMinimizationMethod(theory=BooleanTheory))
            minimize_terms_db(db, method, (mt for mt in terms_db.values() if mt.original not in db),
                              n_jobs=self.n_jobs)
        return db

if __name__ == "__main__":
    exp = BooleanTermMinimizationExperiment("boolean-min-01", n_jobs=6)
    for mterm in exp.naively_minimized_terms(False).values():
        print(term_size(mterm.original), term_size(mterm.minimized))

