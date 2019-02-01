
import itertools
import collections
import attr
import logging

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse, \
    still_match, ByRule, IthElementReason, list_term_elements, leaf_nodes

class Rule:
    def __init__(self, name=None, trigger=None):
        self.name = name
        self.trigger = trigger

    def rewrite(self, match):
        pass

    def __repr__(self):
        return "Rule(" + self.name + ")"

class CongruenceRule(Rule):
    def __init__(self, label, arity):
        self.name = "congruence(" + str(label) + ")"

        lsrc, rsrc = Node(), Node()
        dst = [Node() for _ in range(arity)]
        h1 = Hyperedge(label, lsrc, dst)
        h2 = Hyperedge(label, rsrc, dst)

        self.trigger = (h1, h2)

    def rewrite(self, match):
        return {}

class EqualityRule(Rule):
    def __init__(self, equality, reverse=False, destructive=False, name=None):
        equality = parse(equality)

        if name is None:
            name = ("(rev)" if reverse else "") + ("(des)" if destructive else "") + equality.name

        lhs = equality.lhs
        rhs = equality.rhs

        if reverse:
            lhs, rhs = rhs, lhs

        if isinstance(lhs, Term):
            node, lhs_hyperedge, *_ = list_term_elements(lhs)
            lhs = node
        else:
            lhs_hyperedge = None
            if destructive:
                logging.warning("LHS is a node, destructivity is ignored")
                destructive = False

        if not isinstance(lhs, (Term, Node)) or not isinstance(rhs, (Term, Node)):
            raise ValueError("Both lhs and rhs of the equality should be Terms or nodes: {} = {}"
                             .format(lhs, rhs))

        self.equality = equality
        self.reverse = reverse
        self.destructive = destructive
        self.name = name
        self.trigger = lhs

        self._lhs = lhs
        self._rhs = rhs
        self._lhs_hyperedge = lhs_hyperedge
        self._rhs_leaf_nodes = leaf_nodes(rhs)

    def rewrite(self, match):
        reason = ByRule(self, match)

        # If there are nodes in rhs that has no corresponding match from the lhs, we should create
        # new nodes for them (like x -> and(x, or(x, y)), y is such a node)
        for n in self._rhs_leaf_nodes:
            if not n in match:
                match = dict(match)
                match[n] = Node()

        if isinstance(self._rhs, Term):
            to_add = list_term_elements(self._rhs.apply_map(match),
                                        top_node=match[self._lhs],
                                        reason=reason)
            if self.destructive:
                return {'remove': [match[self._lhs_hyperedge]], 'add': to_add}
            else:
                return {'add': to_add}
        else:
            to_merge = [(match[self._lhs], self._rhs.apply_map(match), reason)]
            if self.destructive:
                return {'remove': [match[self._lhs_hyperedge]], 'merge': to_merge}
            else:
                return {'merge': to_merge}


class Rewriter(Listener):
    def __init__(self, hypergraph, trigger_manager=None, score=None):
        if trigger_manager is None:
            trigger_manager = TriggerManager(hypergraph)

        if score is None:
            score = lambda rule, match, rewrite: 0

        self.score = score

        self.hypergraph = hypergraph
        self.trigger_manager = trigger_manager

        self.pending = []

    def add_rule(self, rule):
        def _callback(match, self=self, rule=rule):
            self.pending.append((rule, match))
        self.trigger_manager.add_trigger(rule.trigger, _callback)

    def perform_rewriting(self, score=None, max_n=None, must_still_match=True):
        if score is None:
            score = self.score

        unapplicable = []
        rule_match_rw_score = []
        for p in self.pending:
            if not must_still_match or still_match(p[1], self.hypergraph):
                rw = p[0].rewrite(p[1])
                if rw is None:
                    # unapplicable for unknown reasons
                    unapplicable.append(p)
                else:
                    sc = score(p[0], p[1], rw)
                    if sc is None:
                        # None score means that the match should be discarded
                        continue
                    else:
                        rule_match_rw_score.append((p[0], p[1], rw, sc))

        rule_match_rw_score.sort(key=lambda tup: tup[3], reverse=True)

        rewrite = {'remove': [], 'add': [], 'merge': []}
        for tup in rule_match_rw_score[:max_n]:
            for field in rewrite:
                rewrite[field].extend(tup[2].get(field, []))

        self.hypergraph.rewrite(**rewrite)

        new_pending = [(rule, match) for rule, match, _, _ in rule_match_rw_score[max_n:]]
        new_pending.extend(unapplicable)
        self.pending = new_pending
