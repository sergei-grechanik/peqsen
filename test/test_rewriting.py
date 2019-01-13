
import peqsen
from peqsen import *

import hypothesis
from hypothesis import given, strategies, reproduce_failure, seed

import sys
import pprint
import attr

@given(strategies.data())
def test_parse(data):
    t1 = data.draw(gen_term())
    t2 = parse(str(t1))
    h1 = Hypergraph()
    h2 = Hypergraph()
    h1.rewrite(add=t1)
    h2.rewrite(add=t2)
    assert h1.isomorphic(h2)

@given(strategies.data())
def test_list_elements(data):
    t1 = data.draw(gen_term())
    t2 = list_term_elements(t1)
    h1 = Hypergraph()
    h2 = Hypergraph()
    h1.rewrite(add=t1)
    h2.rewrite(add=t2)
    assert h1.isomorphic(h2)

@given(strategies.data())
def test_parse_eq(data):
    (eq1_lhs, eq1_rhs) = data.draw(gen_term(equality=True))
    eq2 = parse(str(eq1_lhs) + " = " + str(eq1_rhs))
    h1 = Hypergraph()
    h2 = Hypergraph()
    na1, nb1, *_ = h1.rewrite(add=[eq1_lhs, eq1_rhs])
    h1.rewrite(merge=[(na1, nb1)])
    na2, nb2, *_ = h2.rewrite(add=[eq2.lhs, eq2.rhs])
    h2.rewrite(merge=[(na2, nb2)])
    assert h1.isomorphic(h2)

def check_explanation(data, graph, explanator):
    multinodes = [n for n in graph.nodes() if len(n.outgoing) + len(n.incoming) > 1]

    if multinodes:
        node = data.draw(strategies.sampled_from(multinodes))
        inc_list = sorted(IncidentNode.all_for_node(node))
        inc1 = data.draw(strategies.sampled_from(inc_list))
        inc_list.remove(inc1)
        inc2 = data.draw(strategies.sampled_from(inc_list))
        h1 = inc1.hyperedge
        h2 = inc2.hyperedge
        script = explanator.script([h1, h2, (inc1, inc2)])
        print("Chosen")
        print(h1)
        print(h2)
        print("script")
        print(dump_script(script))
        print()

        new_graph = Hypergraph()
        hh1, hh2, *_ = run_script(new_graph, script)
        hh1 = hh1.follow()
        hh2 = hh2.follow()
        print("New graph:")
        print(new_graph)
        print()
        assert hh1.label == h1.label and hh2.label == h2.label
        assert hh1.incident(inc1.index) == hh2.incident(inc2.index)
    else:
        print("No multinode")
        pass

def check_theory(data, theory, evaluablesig=None):
    graph = Hypergraph()
    rewriter = Rewriter(graph)
    num_models = data.draw(strategies.integers(1, 20))
    evaluator = SimpleEvaluator(graph, evaluablesig, data=data, num_models=num_models)
    explanator = ExplanationTracker(graph)

    for e in theory.equalities:
        destr = False #data.draw(strategies.booleans())
        if data.draw(strategies.booleans()):
            rewriter.add_rule(equality_to_rule(e, destructive=destr))
        if data.draw(strategies.booleans()):
            rewriter.add_rule(equality_to_rule(e, destructive=destr, reverse=True))

    vars_number = data.draw(strategies.integers(0, 4))
    variables = [Node() for _ in range(vars_number)]
    terms = data.draw(strategies.lists(gen_term(theory.signature, variables=variables), max_size=5))
    graph.rewrite(add=terms)

    print("===============================================")
    print()
    print("Terms:")
    for t in terms: print(t)
    print()
    print("Initial graph")
    print(graph)
    print()

    for i in range(data.draw(strategies.integers(1, 20))):
        rewriter.perform_rewriting(max_n=10)

    print()
    print("Rewritten graph")
    print(graph)
    print()

    check_explanation(data, graph, explanator)

@given(strategies.data())
def test_boolean(data):
    check_theory(data, BooleanTheory, BooleanSig)

@given(strategies.data())
def test_boolean_ext(data):
    check_theory(data, BooleanExtTheory, BooleanExtSig)

@given(strategies.data())
def test_integer(data):
    check_theory(data, IntegerTheory, IntegerSig)

if __name__ == "__main__":
    test_parse()
    test_list_elements()
    test_parse_eq()
    test_boolean()
    test_boolean_ext()
    test_integer()
