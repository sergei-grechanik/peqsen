
import itertools
import collections
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term

Multerm = collections.namedtuple('Multerm', 'terms')
"""Used internally in TriggerManager. Multerm is a list of terms that can be
matched against a node, meaning that the node must have hyperedges matching these terms."""

NodeMatches = collections.namedtuple('NodeMatches', 'node terms')
"""Used internally in TriggerManager. NodeMatches is a (set of) matching of a node against a
Multerm, terms must be a list of lists of HyperedgeMatches (that is, each term may match several
hyperedges, so this represents a collection of matches)"""

HyperedgeMatches = collections.namedtuple('HyperedgeMatches', 'hyperedge dst')
"""Used internally in TriggerManager. HyperedgeMatches is a matching of a hyperedge against a Term,
dst must be a list of NodeMatches"""

class TriggerManager(Listener):
    def __init__(self, hypergraph):
        if hypergraph:
            self.hypergraph = hypergraph
            hypergraph.listeners.add(self)

        self._node_to_multerms_to_edgesetlists = {}
        self._multerm_callbacks = {}
        self._merge_callbacks = {}
        self._multerms_to_parents = {}
        self._hyperedge_callbacks = {}

    def add_trigger(self, pattern, callback):
        assert isinstance(pattern, Node)
        assert len(pattern.outgoing) > 0

        multerm, matchlist = self._pattern_to_multerm(pattern)
        self._add_multerm(multerm)

        pat_to_index = {}
        index_mergelist = []
        for i, e in enumerate(matchlist):
            if e in pat_to_index:
                if isinstance(e, Node):
                    index_mergelist.append((pat_to_index[e], i))
            else:
                pat_to_index[e] = i

        def _on_this_multerm(matches, pat_to_index=pat_to_index, index_mergelist=index_mergelist,
                             self=self, callback=callback):
            for matched in self._matches_to_matchlists(matches):
                need_merged = [(matched[p[0]], matched[p[1]]) for p in index_mergelist]

                def _on_pattern_matched(matched=matched, pat_to_index=pat_to_index,
                                        self=self, callback=callback):
                    match = {n: matched[i].follow() for n, i in pat_to_index.items()}
                    if all(e in self.hypergraph for e in match.values()):
                        callback(match)

                self._add_multimerge_callback(need_merged, _on_pattern_matched)

        self._multerm_callbacks.setdefault(multerm, []).append(_on_this_multerm)

    def _pattern_to_multerm(self, pattern):
        matchlist = [pattern]
        subterms = []
        if isinstance(pattern, Node):
            for h in pattern.outgoing:
                mterm, mlst = self._pattern_to_multerm(h)
                matchlist.extend(mlst)
                subterms.append(mterm)
            return Multerm(tuple(sorted(subterms))), matchlist
        elif isinstance(pattern, Hyperedge):
            for d in pattern.dst:
                mterm, mlst = self._pattern_to_multerm(d)
                matchlist.extend(mlst)
                subterms.append(mterm)
            return Term(pattern.label, subterms), matchlist

    def _matches_to_matchlists(self, matches):
        if isinstance(matches, NodeMatches):
            dst_matchlists = [itertools.chain(*(self._matches_to_matchlists(m) for m in ms))
                              for ms in matches.terms]
            for dst in itertools.product(*dst_matchlists):
                yield [matches.node] + list(itertools.chain(*dst))
        elif isinstance(matches, HyperedgeMatches):
            dst_matchlists = [self._matches_to_matchlists(m) for m in matches.dst]
            for dst in itertools.product(*dst_matchlists):
                yield [matches.hyperedge] + list(itertools.chain(*dst))

    def _add_multimerge_callback(self, need_merged, callback):
        while need_merged:
            n1, n2 = need_merged.pop()
            if n1.follow() != n2.follow():
                if n1 > n2:
                    n1, n2 = n2, n1
                def _new_callback(self=self, need_merged=need_merged, callback=callback):
                    self._add_multimerge_callback(need_merged, callback)
                self._merge_callbacks.setdefault((n1, n2), []).append(_new_callback)
                return
        callback()

    def _add_multerm(self, multerm):
        if multerm not in self._multerms_to_parents:
            self._multerms_to_parents[multerm] = []
            for h_idx, h in enumerate(multerm.terms):
                for d_idx, d in enumerate(h.dst):
                    self._add_multerm(d)
                    self._multerms_to_parents[d].append((d, h_idx, d_idx))

                def _check(hyperedge, self=self, multerm=multerm, h_idx=h_idx):
                    self._check_hyperedge_multerm(hyperedge, multerm, h_idx)

                self._hyperedge_callbacks.setdefault((h.label, len(h.dst)), []).append(_check)

    def _check_hyperedge_multerm(self, hyperedge, multerm, h_idx):
        matches = self._get_hyperedge_matches(hyperedge, multerm.terms[h_idx])

        if matches is None:
            return

        self._on_add_hyperedge_matches(multerm, h_idx, matches)

    def _get_hyperedge_matches(self, hyperedge, multerm_hyperedge):
        submatches = []
        for submulterm, d in zip(multerm_hyperedge.dst, hyperedge.dst):
            submatch = self._get_node_matches(d, submulterm)
            if submatch is not None:
                submatches.append(submatch)
            else:
                return None
        return HyperedgeMatches(hyperedge, submatches)

    def _get_node_matches(self, node, multerm):
        edgesetlist = self._node_to_multerms_to_edgesetlists.get(node, {}).get(multerm)
        if edgesetlist is not None and all(s for s in edgesetlist):
            return NodeMatches(node, [(self._get_hyperedge_matches(h, ph) for h in s)
                                      for ph, s in zip(multerm.terms, edgesetlist)])
        else:
            return None

    def _on_add_node_matches(self, multerm, matches):
        for cb in self._multerm_callbacks.get(multerm, []):
            cb(matches)
        for pmult, h_idx, d_idx in self._multerms_to_parents[multerm]:
            for h in matches.node.incoming:
                if h.dst[d_idx] == matches.node and h.label == pmult.terms[h_idx].label:
                    h_matches = self._get_hyperedge_matches(h, pmult.terms[h_idx])
                    if h_matches:
                        h_matches[1][d_idx] = matches
                        self._on_add_hyperedge_matches(pmult, h_idx,
                                                       HyperedgeMatches(h, h_matches))

    def _on_add_hyperedge_matches(self, multerm, h_idx, matches):
        hyperedge = matches.hyperedge

        src_edgesetlist = \
            self._node_to_multerms_to_edgesetlists\
            .setdefault(hyperedge.src, {})\
            .setdefault(multerm, [set() for s in multerm.terms])
        src_edgesetlist[h_idx].add(hyperedge)

        new_matches = []
        for i, (ph, s) in enumerate(zip(multerm.terms, src_edgesetlist)):
            if i == h_idx:
                new_matches.append([matches])
            else:
                if s:
                    new_matches.append([self._get_hyperedge_matches(h, ph) for h in s])
                else:
                    return

        self._on_add_node_matches(multerm, NodeMatches(hyperedge.src, new_matches))

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Hyperedge):
                for c in self._hyperedge_callbacks.get((e.label, len(e.dst)), []):
                    c(e)

    def on_merge(self, hypergraph, node, removed, added):
        pair = tuple(sorted((node, node.merged)))
        callbacks = self._merge_callbacks.get(pair)
        if callbacks is not None:
            for cb in callbacks:
                cb()
            del self._merge_callbacks[pair]

        mult_to_edgeslist = self._node_to_multerms_to_edgesetlists.get(node)
        if mult_to_edgeslist is not None:
            for multerm, edgesetlist in mult_to_edgeslist.items():
                new_edgesetlist = \
                    self._node_to_multerms_to_edgesetlists\
                    .setdefault(node.merged, {})\
                    .setdefault(multerm, [set() for s in multerm.terms])
                was_unfull = False
                became_full = True
                for i in range(len(edgesetlist)):
                    if not new_edgesetlist[i]:
                        was_unfull = True
                        if not edgesetlist[i]:
                            became_full = False
                    new_edgesetlist[i].update(h.follow() for h in edgesetlist[i])
                if was_unfull and became_full:
                    node_matches = NodeMatches(node.merged,
                                               [(self._get_hyperedge_matches(h, ph) for h in s)
                                                for ph, s in zip(multerm.terms, edgesetlist)])
                    self._on_add_node_matches(multerm, node_matches)

            del self._node_to_multerms_to_edgesetlists[node]

    def on_remove(self, hypergraph, elements):
        pass
    def on_remove_node(self, hypergraph, node, hyperedges):
        pass

def eq_label_matcher(pattern_label, match_label):
    return pattern_label == match_label

def combine_submatches(*args):
    res = {}
    for m in args:
        for k, v in m.items():
            if res.setdefault(k, v) != v:
                return None
    return res

def find_matches_hyperedge(hyperedge, pattern, label_matcher):
    if label_matcher(pattern.label, hyperedge.label) and len(hyperedge.dst) == len(pattern.dst):
        submatches = [find_matches_node(d, p, label_matcher)
                      for d, p in zip(hyperedge.dst, pattern.dst)]
        for sub in itertools.product(*submatches):
            combined = combine_submatches(*sub)
            if combined is not None:
                combined[pattern] = hyperedge
                yield combined

def find_matches_node(node, pattern, label_matcher):
    submatches = [itertools.chain(*(find_matches_hyperedge(h, p, label_matcher)
                                    for h in node.outgoing))
                  for p in pattern.outgoing]
    for sub in itertools.product(*submatches):
        combined = combine_submatches(*sub)
        if combined is not None:
            combined[pattern] = node
            yield combined

def find_matches(hypergraph, pattern, label_matcher=eq_label_matcher):
    for n in hypergraph.nodes():
        for m in find_matches_node(n, pattern, label_matcher):
            yield m
