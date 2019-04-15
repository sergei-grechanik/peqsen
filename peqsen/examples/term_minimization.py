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
import math

from typing import Optional, Tuple, Union, Dict, Any, Iterable

tqdm_impl = tqdm.tqdm

class FakePool:
    def imap_unordered(self, fun, args):
        return (fun(a) for a in args)

def make_multiprocessing_pool(n_jobs):
    if n_jobs is None:
        return FakePool()
    else:
        return multiprocessing.Pool(n_jobs)

@attr.s()
class MinimizedTerm:
    original: Term = attr.ib()
    minimized: Term = attr.ib()
    script: Optional[Script] = attr.ib(default=None)
    method: Optional['MinimizationMethod'] = attr.ib(default=None)
    base: Optional['MinimizedTerm'] = attr.ib(default=None)
    stats: Dict = attr.ib(factory=dict)

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
    def minimize(self, term: Term, script: Optional[Script]=None) -> Tuple[Term, Script, Dict]:
        """Return the minimized term, the script to perform minimization and some dict
        representing additional statistics"""
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
        suitable_labels = [(l, a) for l, a in signature.items()
                           if a + 1 <= size and a > 0]
        if not suitable_labels:
            suitable_labels = [(l, a) for l, a in signature.items() if a > 0]
        label, arity = random.choice(suitable_labels)
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

    pool = make_multiprocessing_pool(n_jobs)
    args = [(signature, i, random.randint(0, sys.maxsize))
            for i in range(min_size, max_size) for _ in range(count_per_size)]
    desc = "Generating random terms size in [{}, {}]".format(min_size, max_size)
    for t in tqdm_impl(pool.imap_unordered(_generate_term, args), desc=desc, total=len(args)):
        db[t] = db.get(t, MinimizedTerm(t, t))

def _minimize_term(arg):
    minimizer, mterm, seed = arg
    random.seed(seed)
    try:
        new_term, script, stats = minimizer.minimize(mterm.original, mterm.script)
        new_mterm = MinimizedTerm(mterm.original, new_term, script=script,
                                  base=mterm, method=minimizer, stats=stats)
        return new_mterm
    except Exception as e:
        raise e
        return (e, arg)

def minimize_terms_db(db: MinimizedTermDB,
                      minimizer: MinimizationMethod,
                      minimized_terms: Iterable[MinimizedTerm],
                      *, n_jobs=4, seed=42):
    random.seed(seed)

    pool = make_multiprocessing_pool(n_jobs)
    args = [(minimizer, mterm, random.randint(0, sys.maxsize))
            for mterm in minimized_terms]
    desc = "Minimizing terms with {}".format(type(minimizer).__name__)
    for mt in tqdm_impl(pool.imap_unordered(_minimize_term, args), desc=desc, total=len(args)):
        if isinstance(mt, tuple) and isinstance(mt[0], Exception):
            ex, (_, mterm, _) = mt
            print()
            print("Exception happened during term minimization:")
            print("term", mterm.original)
            print(mterm)
            print(ex)
            print()
            raise ex
        else:
            previous = db.get(mt.original)
            if previous is not None:
                db[mt.original] = min(previous, mt)
            else:
                db[mt.original] = mt

class ScoringMethod:
    def __init__(self, graph):
        self.graph = graph
    def __call__(self, rule, match):
        return 0

class AverageMinSizeScoringMethod(ScoringMethod):
    def __init__(self, graph):
        self.smallest_tracker = graph.get_listener(SmallestHyperedgeTracker)
    def __call__(self, rule, match):
        score = 0
        nodes = 0
        for n in match.values():
            if isinstance(n, Node):
                nodes += 1
                score += self.smallest_tracker.smallest_size(n)
        if nodes == 0:
            return 0
        return score/nodes

@attr.s()
class ScoreMinimizationMethod:
    theory = attr.ib()
    scoring_method = attr.ib()
    max_steps: int = attr.ib(default=1000)

    def minimize(self, term: Term, script: Optional[Script]=None,
                 max_steps=None) -> Tuple[Term, Script, Dict]:
        graph = Hypergraph()
        rewriter = Rewriter(graph)
        explanator = ExplanationTracker(graph)
        smallest_tracker = SmallestHyperedgeTracker(graph)
        scoring_function = self.scoring_method(graph)

        stats = {}
        stats['min_size_dynamics'] = []
        stats['pending_matches_dynamics'] = []
        stats['added_counter_dynamics'] = []
        stats['discarded_counter_dynamics'] = []

        for e in self.theory.equalities:
            rewriter.add_rule(EqualityRule(e))
            rewriter.add_rule(EqualityRule(e, reverse=True))

        added_elements = graph.rewrite(add=term)[1:]
        term_node = added_elements[0]

        top = Node()
        original_match = \
            {k: v for k, v in zip(list_term_elements(term, top_node=top), added_elements)}

        if max_steps is None:
            max_steps = self.max_steps

        if script is not None:
            run_script(graph, script)
            start_step = script_length(script)
        else:
            start_step = 0

        for i in range(start_step, max_steps):
            rewriter.perform_rewriting(max_n=1, score=scoring_function)
            current_min_size = smallest_tracker.smallest_size(term_node.follow())
            stats['min_size_dynamics'].append((i, current_min_size))
            total_pending = len(rewriter.pending_scored) + len(rewriter.pending_unscored)
            stats['pending_matches_dynamics'].append((i, total_pending))
            stats['added_counter_dynamics'].append((i, rewriter.added_counter))
            stats['discarded_counter_dynamics'].append((i, rewriter.discarded_counter))

        smallest_term, smallest_match = \
            smallest_tracker.smallest_term_match_single(term_node.follow())
        smallest_match.update(original_match)
        script = explanator.script(smallest_match)

        return smallest_term, script, stats

@attr.s()
class NaiveMinimizationMethod:
    theory = attr.ib()
    max_steps: int = attr.ib(default=1000)

    def minimize(self, term: Term, script: Optional[Script]=None,
                 max_steps=None) -> Tuple[Term, Script, Dict]:
        m = ScoreMinimizationMethod(theory=self.theory, max_steps=self.max_steps,
                                    scoring_method=ScoringMethod)
        return m.minimize(term, script, max_steps)

@attr.s()
class OneLayerMinimizationMethod:
    theory = attr.ib()
    max_steps: int = attr.ib(default=None)

    def minimize(self, term: Term, script: Optional[Script]=None,
                 max_steps=None) -> Tuple[Term, Script, Dict]:
        graph = Hypergraph()
        rewriter = Rewriter(graph)
        explanator = ExplanationTracker(graph)
        smallest_tracker = SmallestHyperedgeTracker(graph)

        stats = {}
        stats['min_size_dynamics'] = []
        stats['pending_matches_dynamics'] = []

        for e in self.theory.equalities:
            rewriter.add_rule(EqualityRule(e))
        for e in self.theory.equalities:
            rewriter.add_rule(EqualityRule(e, reverse=True))

        added_elements = graph.rewrite(add=term)[1:]
        term_node = added_elements[0]

        top = Node()
        original_match = \
            {k: v for k, v in zip(list_term_elements(term, top_node=top), added_elements)}

        if max_steps is None:
            max_steps = self.max_steps

        if script is not None:
            run_script(graph, script)
            start_step = script_length(script)
        else:
            start_step = 0

        # Don't do the next layer
        graph.listeners.discard(rewriter.trigger_manager)

        scoring_function = lambda *_: random.random()

        # rewriter.perform_rewriting(max_n=max_steps)
        for i in range(start_step, max_steps):
            rewriter.perform_rewriting(max_n=1, score=scoring_function)
            current_min_size = smallest_tracker.smallest_size(term_node.follow())
            stats['min_size_dynamics'].append((i, current_min_size))
            total_pending = len(rewriter.pending_scored) + len(rewriter.pending_unscored)
            stats['pending_matches_dynamics'].append((i, total_pending))

        smallest_term, smallest_match = \
            smallest_tracker.smallest_term_match_single(term_node.follow())
        smallest_match.update(original_match)
        script = explanator.script(smallest_match)

        return smallest_term, script, stats

@attr.s()
class RepeatedMinimizationMethod:
    method: MinimizationMethod = attr.ib()
    iterations: int = attr.ib(default=10)

    def minimize(self, term: Term, script: Optional[Script]=None) -> Tuple[Term, Script, Dict]:
        best_term = term
        best_script = script
        stats_list = []

        for i in range(self.iterations):
            new_term, new_script, stats = \
                self.method.minimize(best_term)
            stats_list.append(stats)
            if best_script is None:
                best_script = new_script
            else:
                best_script = compose_scripts(best_script, new_script, term=best_term)
            best_term = new_term

        return best_term, best_script, {'substats': stats_list}

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

