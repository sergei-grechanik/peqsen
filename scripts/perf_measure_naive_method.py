
# PYTHONPATH=".:$PYTHONPATH" python -m cProfile -o _prof_result ./scripts/perf_measure_naive_method.py
# pyprof2calltree -i _prof_result
# kcachegrind _prof_result.log

from peqsen.examples.term_minimization import *

max_size = 11
min_size = 10
count_per_size = 5
max_steps = 200
variables_num = 3
n_jobs = None

db = {}
generate_random_terms_db(db, BooleanSig.signature, n_jobs=n_jobs,
                         min_size=min_size, max_size=max_size, count_per_size=count_per_size,
                         variables=variables_num)

db_naively_minimized = {}
method = NaiveMinimizationMethod(theory=BooleanTheory, max_steps=max_steps)
minimize_terms_db(db_naively_minimized, method, db.values(), n_jobs=n_jobs)
