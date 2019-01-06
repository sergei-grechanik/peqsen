
import itertools
import collections
import attr

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse, \
    still_match, ByRule, IthHyperedgeReason, list_term_elements

@attr.s(slots=True, frozen=True, cmp=False)
class Rule:
    name = attr.ib()
    trigger = attr.ib()
    rewrite = attr.ib()

@attr.s(slots=True, frozen=True, cmp=False)
class CongruenceRule:
    name = attr.ib(default="congruence")
    trigger = attr.ib(default=None)
    rewrite = attr.ib(default=lambda _: {})

def equality_to_rule(equality, reverse=False, destructive=False, name=None):
    equality = parse(equality)
    lhs = equality.lhs
    rhs = equality.rhs

    if not isinstance(lhs, (Term, Node)) or not isinstance(rhs, (Term, Node)):
        raise ValueError("Both lhs and rhs of the equality should be Terms or nodes: {} = {}"
                         .format(lhs, rhs))

    if reverse:
        lhs, rhs = rhs, lhs

    if name is None:
        name = ("(rev)" if reverse else "") + ("(des)" if destructive else "") + equality.name

    if isinstance(lhs, Term):
        node, lhs_hyperedge, *_ = list_term_elements(lhs)
        lhs = node
    else:
        lhs_hyperedge = None
        assert not destructive, "Not supported"

    rule_container = []

    def _rewrite(match, lhs_hyperedge=lhs_hyperedge, rhs=rhs,
                 destructive=destructive, rule_container=rule_container):
        reason = ByRule(rule_container[0], match)
        to_add = list_term_elements(rhs.apply_map(match))
        to_add = [e.with_reason(IthHyperedgeReason(reason, i))
                  for i, e in enumerate(to_add)]
        if destructive:
            return {'remove': [match[lhs_hyperedge]], 'add': to_add}
        else:
            return {'add': to_add}

    rule_container.append(Rule(name=name, trigger=lhs, rewrite=_rewrite))

    return rule_container[0]

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
