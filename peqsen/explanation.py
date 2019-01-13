
import itertools
import collections
import attr

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse, \
    still_match, match_follow, ByRule, ByCongruence, IthElementReason, Rule, CongruenceRule, \
    AddTermReason, list_term_elements, some_name, GloballyIndexed

from peqsen import util

@attr.s(slots=True, frozen=True)
class IncidentNode:
    """An incident node/point of a hyperedge or a representation of a hyperedge. Index -1 means the
    source node, index >= 0 means the dst index."""
    hyperedge = attr.ib()
    index = attr.ib(default=-1)

    @staticmethod
    def first_of(hyperedge, node):
        if hyperedge.src == node:
            return IncidentNode(hyperedge, -1)
        else:
            return IncidentNode(hyperedge, hyperedge.dst.index(node))

    @staticmethod
    def all_of(hyperedge, node):
        res = []
        if hyperedge.src == node:
            res.append(IncidentNode(hyperedge, -1))

        for i, d in enumerate(hyperedge.dst):
            if d == node:
                res.append(IncidentNode(hyperedge, i))

        return res

    @staticmethod
    def all_for_node(node):
        res = []
        for h in node.outgoing:
            res.append(IncidentNode(h, -1))
        for h in node.incoming:
            res.extend(IncidentNode.all_of(h, node))
        return res

# These are compared by reference
@attr.s(frozen=True, cmp=False)
class FreeNodeReason:
    pass

# These are compared by reference
@attr.s(frozen=True, cmp=False)
class MatchScript:
    elements = attr.ib()
    hyperedge_scripts = attr.ib()
    merge_scripts = attr.ib()
    node_scripts = attr.ib()

# These are compared by reference
@attr.s(frozen=True, cmp=False)
class RuleApplicationScript:
    rule = attr.ib()
    match_script = attr.ib()

@attr.s(frozen=True)
class IthElementScript:
    script = attr.ib()
    index = attr.ib()

@attr.s(frozen=True)
class RunAllScript:
    subscripts = attr.ib()

@attr.s(frozen=True)
class AddTermScript:
    term = attr.ib()

# These are compared by reference
@attr.s(frozen=True, cmp=False)
class FreeNodeScript:
    pass

def run_script(hypergraph, script, cache=None):
    if cache is None:
        cache = {}

    if isinstance(script, RunAllScript):
        return [run_script(hypergraph, s, cache) for s in script.subscripts]
    elif isinstance(script, IthElementScript):
        return run_script(hypergraph, script.script, cache)[script.index]
    elif isinstance(script, IncidentNode):
        return run_script(hypergraph, script.hyperedge, cache).incident(script.index)
    elif script in cache:
        return cache[script]
    elif isinstance(script, FreeNodeScript):
        added = hypergraph.rewrite(add=Node())[0]
        cache[script] = added
        return added
    elif isinstance(script, MatchScript):
        match = {}
        elements = script.elements
        for e in elements:
            if isinstance(e, Hyperedge):
                match[e] = run_script(hypergraph, script.hyperedge_scripts[e], cache)

        for in1, in2, ms in script.merge_scripts:
            n1 = match[in1.hyperedge].incident(in1.index).follow()
            n2 = match[in2.hyperedge].incident(in2.index).follow()
            if n1 != n2:
                run_script(hypergraph, ms, cache)

        for e in elements:
            if isinstance(e, Node):
                if not e in match:
                    incident_hyperedges = list(e.outgoing) + list(e.incoming)
                    if incident_hyperedges:
                        h = incident_hyperedges[0]
                        idx = IncidentNode.first_of(h, e).index
                        match[e] = match[h].incident(idx)
                    else:
                        match[e] = run_script(hypergraph, script.node_scripts[e], cache)

        match = match_follow(match)

        if not still_match(match, hypergraph):
            raise RuntimeError("The match built according to the script "
                               "doesn't match the hypergraph")

        cache[script] = match
        return match
    elif isinstance(script, RuleApplicationScript):
        match = run_script(hypergraph, script.match_script, cache)
        match = match_follow(match)

        rw = script.rule.rewrite(match)
        added = hypergraph.rewrite(**rw)
        print()
        print("Applying rule", script.rule)
        print("Match", match)
        print(hypergraph)

        cache[script] = added
        return added
    elif isinstance(script, AddTermScript):
        added = hypergraph.rewrite(add=list_term_elements(script.term))
        print()
        print("Adding term", script.term)
        print(hypergraph)
        cache[script] = added
        return added
    else:
        raise ValueError("Don't know how to runs script {}".format(script))

def make_congruence_rule_and_match(lhs, rhs):
    if lhs.label != rhs.label or lhs.dst != rhs.dst:
        raise ValueError("The hyperedges cannot be congruent")

    lsrc, rsrc = Node(), Node()
    dst = [Node() for _ in lhs.dst]
    h1 = Hyperedge(lhs.label, lsrc, dst)
    h2 = Hyperedge(rhs.label, rsrc, dst)

    match = {lsrc: lhs.src, rsrc: rhs.src, h1: lhs, h2: rhs}
    for d, ld in zip(dst, lhs.dst):
        match[d] = ld

    rule = CongruenceRule()

    return (rule, match)

class Point(GloballyIndexed):
    def __init__(self, merged=None, merge_reason=None):
        super().__init__()
        self.merged = merged
        self.merge_reason = merge_reason

    def detail(self):
        points = []
        reasons = ""
        point = self
        while point is not None:
            points.append(some_name(point))
            if point.merge_reason is not None:
                reasons += "\n  -> " + some_name(point.merged) + "  " + repr(point.merge_reason)
            point = point.merged
        return "Point(" + "->".join(points) + "):" + reasons + "\n"

    def __repr__(self):
        points = []
        point = self
        while point is not None:
            points.append(some_name(point))
            point = point.merged
        return "Point(" + "->".join(points) + ")"

@util.for_all_methods(util.traced)
class ExplanationTracker(Listener):
    def __init__(self, hypergraph):
        hypergraph.listeners.add(self)
        self.node_points = {}
        self.hyperedge_points = {}
        self.node_first_hyperedges = {}

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Node):
                self.node_points[e] = Point()
                print()
                print("Added node", e, self.node_points[e])
                print()
        for e in elements:
            if isinstance(e, Hyperedge):
                src_point = Point(merged=self.node_points[e.src], merge_reason=e.reason)
                dst_points = [Point(merged=self.node_points[d], merge_reason=e.reason)
                              for i, d in enumerate(e.dst)]
                self.hyperedge_points[e] = [(e.reason, src_point, dst_points)]
                print()
                print("Added hyperedge", e)
                print("  reason", e.reason)
                print("  points", src_point, dst_points)
                print()
        for e in elements:
            if isinstance(e, Node):
                if e not in self.node_first_hyperedges:
                    incs = IncidentNode.all_for_node(e)
                    if incs:
                        self.node_first_hyperedges[e] = incs[0]
                    else:
                        self.node_first_hyperedges[e] = FreeNodeReason()
                    print()
                    print("Node_reason", e, self.node_first_hyperedges[e])
                    print()

        self.dump()

    def on_merge(self, hypergraph, node, removed, added, reason):
        p1 = self.node_points[node]
        p2 = self.node_points[node.merged]
        p = Point()

        p1.merged = p
        p2.merged = p
        p1.merge_reason = reason
        p2.merge_reason = reason

        print()
        print("Merged")
        print(p1.detail())
        print(p2.detail())
        print()

        self.node_points[node.merged] = p
        del self.node_points[node]

        for h in removed:
            self.hyperedge_points.setdefault(h.merged, []).extend(self.hyperedge_points.get(h, []))

        self.dump()

    def get_points(self, incident):
        return [srcp if incident.index == -1 else dstp[incident.index]
                for (_, srcp, dstp) in self.hyperedge_points[incident.hyperedge]]

    def join_points(self, point1, point2):
        path1 = [point1]
        path2 = [point2]

        while True:
            if point1 in path2:
                return (point1, path1, path2[:path2.index(point1) + 1])
            elif point2 in path1:
                return (point2, path1[:path1.index(point2) + 1], path2)
            else:
                if point1 is not None: point1 = point1.merged
                if point2 is not None: point2 = point2.merged
                if point1 is not None: path1.append(point1)
                if point2 is not None: path2.append(point2)
                if point1 is None and point2 is None:
                    raise RuntimeError("Cannot join points {} and {}".format(point1, point2))

    def script_for_merge(self, incident1, incident2, cache):
        pair = (incident1, incident2)

        if pair in cache:
            return cache[pair]

        points1 = self.get_points(incident1)
        points2 = self.get_points(incident2)

        # We must use the first set of points and reasons, because it must agree with the reason
        # used for the corresponding hyperedge which is always the first one
        # TODO: Check if the hyperedge was already processed and if it wasn't, try to find the best
        best_paths = self.join_points(points1[0], points2[0])
        #  best_paths = min((self.join_points(p1, p2) for p1 in points1 for p2 in points2),
        #                   key=lambda triple: len(triple[1]) + len(triple[2]))

        assert best_paths[1][-1] == best_paths[0]
        assert best_paths[2][-1] == best_paths[0]

        print("====================")
        print("best_paths[0]", best_paths[0])
        for path in best_paths[1:]:
            for p in path:
                print(p)
            print("---------------")
        print("====================")

        subscripts = []
        for path in best_paths[1:]:
            for p in path[:-1]:
                sc = self.script_for_reason(p.merge_reason, cache)
                util.printind()
                util.printind("Mergescript", p)
                util.printind("Script", sc)
                util.printind()
                subscripts.append(sc)

        res = RunAllScript(subscripts)
        cache[pair] = res
        return res

    def script_for_reason(self, reason, cache):
        if reason in cache:
            res = cache[reason]
            if res is None:
                raise ValueError("Recursion detected while building an explanation: {}"
                                 .format(reason))
            return cache[reason]

        cache[reason] = None

        if isinstance(reason, ByRule):
            match = match_follow(reason.match)
            res = RuleApplicationScript(reason.rule, self.script_for_match(match, cache))
        elif isinstance(reason, ByCongruence):
            rule, match = make_congruence_rule_and_match(reason.lhs.follow(), reason.rhs.follow())
            res = RuleApplicationScript(rule, self.script_for_match(match, cache))
        elif isinstance(reason, AddTermReason):
            res = AddTermScript(reason.term)
        elif isinstance(reason, IthElementReason):
            res = IthElementScript(self.script_for_reason(reason.reason, cache), reason.index)
        elif isinstance(reason, IncidentNode):
            res = IncidentNode(self.script_for_hyperedge(reason.hyperedge, cache), reason.index)
        elif isinstance(reason, FreeNodeReason):
            res = FreeNodeScript()
        else:
            raise RuntimeError("Unsupported reason: {}".format(reason))

        cache[reason] = res
        return res

    def script_for_hyperedge(self, hyperedge, cache):
        return self.script_for_reason(hyperedge.reason, cache)

    def script_for_match(self, match, cache):
        hyperedge_scripts = {}
        node_scripts = {}
        merge_scripts = []

        for e, h in match.items():
            if isinstance(e, Hyperedge):
                hyperedge_scripts[e] = self.script_for_hyperedge(h, cache)

        for e, n in match.items():
            if isinstance(e, Node):
                incident_hyperedges = [h
                                       for h in itertools.chain(n.incoming, n.outgoing)
                                       if h in match]
                incidents = [inc
                             for h in incident_hyperedges
                             for inc in IncidentNode.all_of(h, e)]
                incidents_h = [IncidentNode(match[inc.hyperedge], inc.index)
                               for inc in incidents]

                for inc, inc_h in zip(incidents[1:], incidents_h[1:]):
                    s = self.script_for_merge(incidents_h[0], inc_h, cache)
                    merge_scripts.append((incidents[0], inc, s))

                if not incident_hyperedges:
                    # This is a special case: the node has no incident hyperedge within the match,
                    # so we use some outside hyperedge to generate this node
                    if n in self.node_first_hyperedges:
                        node_scripts[e] = \
                            self.script_for_reason(self.node_first_hyperedges[n], cache)

        res = MatchScript(list(match.keys()), hyperedge_scripts, merge_scripts, node_scripts)
        return res

    def script(self, requirements):
        cache = {}
        scripts = []
        for r in requirements:
            if isinstance(r, Hyperedge):
                scripts.append(self.script_for_hyperedge(r, cache))
            elif isinstance(r, tuple):
                scripts.append(self.script_for_merge(r[0], r[1], cache))
        return RunAllScript(scripts)

    def dump(self):
        print()
        for n, p in self.node_points.items():
            print("node", n, p)
        for n, h in self.node_first_hyperedges.items():
            print("node-hyper", n, h)
        for h, lst in self.hyperedge_points.items():
            print("hyper", h)
            for (r, srcp, dstp) in lst:
                print("  reason", r)
                print("  srcp", srcp, "dstp", [p for p in dstp])
        print()
