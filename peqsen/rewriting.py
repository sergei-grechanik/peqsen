
import itertools
import collections
import attr
import logging
import heapq

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse, \
    still_match, ByRule, IthElementReason, list_term_elements, leaf_nodes, match_follow

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
    """
    Important note: by default we score each match only once. This makes everything much more
    efficient.
    """
    def __init__(self, hypergraph, trigger_manager=None, score=None):
        if trigger_manager is None:
            trigger_manager = TriggerManager(hypergraph)

        if score is None:
            score = lambda rule, match: 0

        self.score = score

        self.hypergraph = hypergraph
        self.trigger_manager = trigger_manager

        self.pending_scored = []
        self.pending_unscored = []

        self.discarded_counter = 0
        self.added_counter = 0

    def add_rule(self, rule):
        def _callback(match, self=self, rule=rule):
            self.pending_unscored.append((self.added_counter, rule, match))
            self.added_counter += 1
        self.trigger_manager.add_trigger(rule.trigger, _callback)

    def score_unscored(self, *, score=None, must_still_match=True):
        if score is None:
            score = self.score

        for index, rule, match in self.pending_unscored:
            match = match_follow(match)
            if not must_still_match or still_match(match, self.hypergraph):
                sc = score(rule, match)
                if sc is None:
                    # None score means that the match should be discarded
                    self.discarded_counter += 1
                    continue
                else:
                    heapq.heappush(self.pending_scored, (-sc, index, rule, match))
            else:
                # Discarded due to not matching anymore
                self.discarded_counter += 1

        self.pending_unscored = []

    def perform_rewriting(self, *, score=None, max_n=None, must_still_match=True):
        self.score_unscored(score=score, must_still_match=must_still_match)

        to_perform = []
        while self.pending_scored and (max_n is None or len(to_perform) < max_n):
            _, _, rule, match = heapq.heappop(self.pending_scored)
            match = match_follow(match)
            if not must_still_match or still_match(match, self.hypergraph):
                to_perform.append((rule, match))
            else:
                # Discarded due to not matching anymore
                self.discarded_counter += 1

        rewrite = {'remove': [], 'add': [], 'merge': []}
        for tup in to_perform:
            rw = tup[0].rewrite(tup[1])
            for field in rewrite:
                rewrite[field].extend(rw.get(field, []))

        self.hypergraph.rewrite(**rewrite)
