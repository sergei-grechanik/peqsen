
import itertools
import collections
import attr

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse, \
    still_match, match_follow, ByRule, ByCongruence, IthElementReason, Rule, CongruenceRule, \
    AddTermsReason, MultipleReason, list_term_elements, some_name, GloballyIndexed

from peqsen import util

class Script:
    pass

@attr.s(slots=True, frozen=True)
class IncidentNode(Script):
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
@attr.s(frozen=True, cmp=False, hash=False)
class FreeNodeReason(Script):
    pass

# These are compared by reference
@attr.s(frozen=True, cmp=False, hash=False)
class MatchScript(Script):
    elements = attr.ib()
    hyperedge_scripts = attr.ib()
    merge_scripts = attr.ib()
    node_scripts = attr.ib()

# These are compared by reference
@attr.s(frozen=True, cmp=False, hash=False)
class RuleApplicationScript(Script):
    rule = attr.ib()
    match_script = attr.ib()

@attr.s(frozen=True)
class IthElementScript(Script):
    script = attr.ib()
    index = attr.ib()

@attr.s(frozen=True)
class RunAllScript(Script):
    subscripts = attr.ib()

@attr.s(frozen=True)
class AddTermsScript(Script):
    terms = attr.ib()

# These are compared by reference
@attr.s(frozen=True, cmp=False, hash=False)
class FreeNodeScript(Script):
    pass

def script_length(script):
    """The size of the script in terms of the number of rule application steps"""
    seen_scripts = set()
    result = [0]

    def _process(script, seen_scripts=seen_scripts, result=result):
        if script not in seen_scripts:
            seen_scripts.add(script)
            if isinstance(script, RunAllScript):
                for s in script.subscripts:
                    _process(s)
            elif isinstance(script, IthElementScript):
                _process(script.script)
            elif isinstance(script, IncidentNode):
                _process(script.hyperedge)
            elif isinstance(script, FreeNodeScript):
                pass
            elif isinstance(script, MatchScript):
                for s in script.hyperedge_scripts.values():
                    _process(s)
                for s in script.node_scripts.values():
                    _process(s)
                for _, _, s in script.merge_scripts:
                    _process(s)
            elif isinstance(script, RuleApplicationScript):
                _process(script.match_script)
                if not isinstance(script.rule, CongruenceRule):
                    result[0] += 1
            elif isinstance(script, AddTermsScript):
                pass
            else:
                raise ValueError("Don't know how to process script {}".format(script))

    _process(script)
    return result[0]

def dump_script(script, cache=None):
    if cache is None:
        cache = {}
        code = dump_script(script, cache)
        code += "\nreturn " + str(cache[script])
        return "script {\n" + code + "\n}"

    if script in cache:
        return ""
    elif isinstance(script, RunAllScript):
        codes = [dump_script(s, cache) for s in script.subscripts]
        code = "\n".join(c for c in codes if c != "")
        cache[script] = [cache[s] for s in script.subscripts]
        return code
    elif isinstance(script, IthElementScript):
        code = dump_script(script.script, cache)
        res = cache[script.script]
        if isinstance(res, str):
            cache[script] = res + "[{}]".format(script.index)
        else:
            cache[script] = cache[script.script][script.index]
        return code
    elif isinstance(script, IncidentNode):
        code = dump_script(script.hyperedge, cache)
        if script.index == -1:
            cache[script] = cache[script.hyperedge] + ".src"
        else:
            cache[script] = cache[script.hyperedge] + ".dst[{}]".format(script.index)
        return code
    elif isinstance(script, FreeNodeScript):
        node = some_name(script)
        cache[script] = node
        return node + " = Node()"
    elif isinstance(script, MatchScript):
        codes = []
        match = {}
        elements = script.elements
        for e in script.elements:
            if isinstance(e, Hyperedge):
                codes.append(dump_script(script.hyperedge_scripts[e], cache))
                match[e] = cache[script.hyperedge_scripts[e]]

        for in1, in2, ms in script.merge_scripts:
            codes.append(dump_script(ms, cache))
            codes.append("Merge  {}  and  {}  by {}".format(in1, in2, cache[ms]))

        for e in elements:
            if isinstance(e, Node):
                if e in script.node_scripts:
                    codes.append(dump_script(script.node_scripts[e], cache))
                    match[e] = cache[script.node_scripts[e]]

        cache[script] = match
        return "\n".join(c for c in codes if c != "")
    elif isinstance(script, RuleApplicationScript):
        codes = [dump_script(script.match_script, cache)]
        match = cache[script.match_script]

        someid = some_name(script)

        codes.append("{} = Rule {}\n        Match {}".format(someid, script.rule, match))
        cache[script] = someid
        return "\n".join(c for c in codes if c != "")
    elif isinstance(script, AddTermsScript):
        someid = some_name(script)
        code = "{} = Terms {}".format(someid, script.terms)
        cache[script] = someid
        return code
    else:
        raise ValueError("Don't know how to dump script {}".format(script))

def run_script(hypergraph, script, cache=None):
    if cache is None:
        cache = {}

    if isinstance(script, RunAllScript):
        result = [run_script(hypergraph, s, cache) for s in script.subscripts]
    elif isinstance(script, IthElementScript):
        result = run_script(hypergraph, script.script, cache)[script.index]
    elif isinstance(script, IncidentNode):
        result = run_script(hypergraph, script.hyperedge, cache).incident(script.index)
    elif script in cache:
        result = cache[script]
    elif isinstance(script, FreeNodeScript):
        added = hypergraph.rewrite(add=Node())[0]
        cache[script] = added
        result = added
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
                    if e in script.node_scripts:
                        match[e] = run_script(hypergraph, script.node_scripts[e], cache)
                    else:
                        # As noted somewhere else, there may be no e.outgoing and e.incoming
                        for h in match:
                            if isinstance(h, Hyperedge):
                                incs = IncidentNode.all_of(h, e)
                                if incs:
                                    match[e] = match[h].incident(incs[0].index)
                                    break
                        assert match[e] is not None

        match = match_follow(match)

        if not still_match(match, hypergraph):
            raise RuntimeError("The match built according to the script "
                               "doesn't match the hypergraph")

        cache[script] = match
        result = match
    elif isinstance(script, RuleApplicationScript):
        match = run_script(hypergraph, script.match_script, cache)
        match = match_follow(match)

        rw = script.rule.rewrite(match)
        added = hypergraph.rewrite(**rw)

        cache[script] = added
        result = added
    elif isinstance(script, AddTermsScript):
        added = hypergraph.rewrite(add=script.terms)[len(script.terms):]
        res = []
        start = 0
        for t in script.terms:
            tsize = len(list_term_elements(t))
            res.append(added[start:start + tsize])
            start += tsize
        cache[script] = res
        result = res
    else:
        raise ValueError("Don't know how to runs script {}".format(script))

    return result

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

    rule = CongruenceRule(lhs.label, len(dst))

    return (rule, match)

#@util.for_all_methods(util.traced)
class ExplanationTracker(Listener):
    def __init__(self, hypergraph):
        hypergraph.listeners.add(self)
        self.node_first_hyperedges = {}

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Node):
                if e not in self.node_first_hyperedges:
                    incs = IncidentNode.all_for_node(e)
                    if incs:
                        self.node_first_hyperedges[e] = incs[0]
                    else:
                        self.node_first_hyperedges[e] = FreeNodeReason()

    def on_merge(self, hypergraph, node, removed, added, reason):
        assert node.merge_reason == reason

    def join_nodes(self, node1, node2):
        node1_orig = node1
        node2_orig = node2
        path1 = [node1]
        path2 = [node2]

        while True:
            if node1 in path2:
                return (node1, path1, path2[:path2.index(node1) + 1])
            elif node2 in path1:
                return (node2, path1[:path1.index(node2) + 1], path2)
            else:
                if node1 is not None: node1 = node1.merged
                if node2 is not None: node2 = node2.merged
                if node1 is not None: path1.append(node1)
                if node2 is not None: path2.append(node2)
                if node1 is None and node2 is None:
                    raise RuntimeError("Cannot join nodes {} and {}".format(node1_orig, node2_orig))

    def script_for_merge(self, incident1, incident2, cache):
        pair = (incident1, incident2)

        if pair in cache:
            return cache[pair]

        node1 = incident1.hyperedge.incident(incident1.index)
        node2 = incident2.hyperedge.incident(incident2.index)
        paths = self.join_nodes(node1, node2)

        subscripts = []
        for path in paths[1:]:
            for p in path[:-1]:
                sc = self.script_for_reason(p.merge_reason, cache)
                subscripts.append(sc)

        res = RunAllScript(tuple(subscripts))
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
            res = RuleApplicationScript(reason.rule, self.script_for_match(reason.match, cache))
        elif isinstance(reason, ByCongruence):
            rule, match = make_congruence_rule_and_match(reason.lhs, reason.rhs)
            res = RuleApplicationScript(rule, self.script_for_match(match, cache))
        elif isinstance(reason, AddTermsReason):
            res = AddTermsScript(reason.terms)
        elif isinstance(reason, IthElementReason):
            res = IthElementScript(self.script_for_reason(reason.reason, cache), reason.index)
        elif isinstance(reason, IncidentNode):
            res = IncidentNode(self.script_for_hyperedge(reason.hyperedge, cache), reason.index)
        elif isinstance(reason, FreeNodeReason):
            res = FreeNodeScript()
        elif isinstance(reason, MultipleReason):
            res = RunAllScript(tuple(self.script_for_reason(r, cache) for r in reason.reasons))
        else:
            raise RuntimeError("Unsupported reason: {}".format(reason))

        cache[reason] = res
        return res

    def script_for_hyperedge(self, hyperedge, cache=None):
        if cache is None:
            cache = {}

        script = self.script_for_reason(hyperedge.reason, cache)
        return script

    def script_for_match(self, match, cache=None):
        if cache is None:
            cache = {}

        hyperedge_scripts = {}
        node_scripts = {}
        merge_scripts = []

        for e, h in match.items():
            if isinstance(e, Hyperedge):
                hyperedge_scripts[e] = self.script_for_hyperedge(h, cache)

        for e, n in match.items():
            if isinstance(e, Node):
                # We cannot loop through e.outgoing and e.incoming because e might be a free node,
                # outside of any hypergraph, and not containing this information
                incidents = [inc
                             for h in match if isinstance(h, Hyperedge)
                             for inc in IncidentNode.all_of(h, e)]
                incidents_h = [IncidentNode(match[inc.hyperedge], inc.index)
                               for inc in incidents]

                for inc, inc_h in zip(incidents[1:], incidents_h[1:]):
                    s = self.script_for_merge(incidents_h[0], inc_h, cache)
                    merge_scripts.append((incidents[0], inc, s))

                if not incidents:
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
            elif isinstance(r, dict):
                scripts.append(self.script_for_match(r, cache))
            else:
                raise ValueError("Don't know how create a script for {} of type {}"
                                 .format(r, type(r)))
        return RunAllScript(tuple(scripts))

