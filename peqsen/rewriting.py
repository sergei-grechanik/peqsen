
import itertools
import collections
import attr

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse

Rule = attr.make_class('Rule', ['name', 'trigger', 'rewrite'], frozen=True)

def equality_to_rule(equality, reverse=False, destructive=False, name=None):
    equality = parse(equality)
    lhs = equality.lhs
    rhs = equality.rhs

    if not isinstance(lhs, Term) or not isinstance(rhs, Term):
        raise ValueError("Both lhs and rhs of the equality should be Terms")

    if reverse:
        lhs, rhs = rhs, lhs

    if name is None:
        name = equality.name + (" (rev)" if reverse else "") + (" (des)" if destructive else "")

    def _rewrite(match, lhs=lhs, rhs=rhs, destructive=destructive):
        if destructive:
            return {'remove': [match[lhs.hyperedge]], 'add': [rhs.apply_map(match)]}
        else:
            return {'add': [rhs.apply_map(match)]}

    return Rule(name=name, trigger=lhs, rewrite=_rewrite)

class Rewriter(Listener):
    def __init__(self, hypergraph, trigger_manager=None):
        if trigger_manager is None:
            trigger_manager = TriggerManager(hypergraph)

        self.hypergraph = hypergraph
        self.trigger_manager = trigger_manager

        self.pending = []

    def add_rule(self, rule):
        def _callback(match, self=self, rule=rule):
            self.pending.append((rule, match))
        self.trigger_manager.add_trigger(rule.trigger, _callback)

    def perform_rewriting(score=None, max_n=None):
        pass
