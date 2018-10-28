
import itertools
import collections
import attr
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term

Multerm = attr.make_class('Multerm', ['terms'], frozen=True)
"""Used internally in TriggerManager. Multerm is a list of terms that can be
matched against a node, meaning that the node must have hyperedges matching these terms."""

NodeMatches = attr.make_class('NodeMatches', ['node', 'terms'], frozen=True)
"""Used internally in TriggerManager. NodeMatches is a (set of) matching of a node against a
Multerm, terms must be a list of lists of HyperedgeMatches (that is, each term may match several
hyperedges, so this represents a collection of matches)"""

HyperedgeMatches = attr.make_class('HyperedgeMatches', ['hyperedge', 'dst'], frozen=True)
"""Used internally in TriggerManager. HyperedgeMatches is a matching of a hyperedge against a Term,
dst must be a list of NodeMatches"""

class TriggerManager(Listener):
    def __init__(self, hypergraph):
        if hypergraph:
            self.hypergraph = hypergraph
            hypergraph.listeners.add(self)

        # Node ~> (Multerm ~> [{Hyperedge}])
        self._node_to_multerms_to_edgesetlists = {}
        # Multerm ~> [NodeMatches -> ()]
        self._multerm_callbacks = {}
        # Node ~> (Node ~> [() -> ()])
        self._merge_callbacks = {}
        # Multerm ~> [(Multerm, Int, Int)]
        self._multerms_to_parents = {}
        # (Label, Int) ~> [Hyperedge -> ()]
        self._hyperedge_callbacks = {}

    def add_trigger(self, pattern, callback):
        assert isinstance(pattern, Node)
        assert len(pattern.outgoing) > 0

        multerm, matchlist = self._pattern_to_multerm(pattern)
        self._add_multerm(multerm)

        print("\n================")
        print(pattern)
        print(multerm)
        print(matchlist)
        print("================\n")

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
            print()
            print("Multerm", multerm)
            print("Matched", matches)
            for matched in self._matches_to_matchlists(matches):
                print("    list", matched)
                print("matchlist", matchlist)
                need_merged = [(matched[p[0]], matched[p[1]]) for p in index_mergelist]
                print("need merged", need_merged)

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
            for h in sorted(pattern.outgoing):
                mterm, mlst = self._pattern_to_multerm(h)
                matchlist.extend(mlst)
                subterms.append(mterm)
            return Multerm(tuple(subterms)), matchlist
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
            return [[matches.node] + list(itertools.chain(*dst))
                    for dst in itertools.product(*dst_matchlists)]
        elif isinstance(matches, HyperedgeMatches):
            dst_matchlists = [self._matches_to_matchlists(m) for m in matches.dst]
            return [[matches.hyperedge] + list(itertools.chain(*dst))
                    for dst in itertools.product(*dst_matchlists)]

    def _add_multimerge_callback(self, need_merged, callback):
        while need_merged:
            n1, n2 = need_merged.pop()
            if n1.follow() != n2.follow():
                if n1 > n2:
                    n1, n2 = n2, n1
                def _new_callback(self=self, need_merged=need_merged, callback=callback):
                    self._add_multimerge_callback(need_merged, callback)
                self._merge_callbacks.setdefault(n1, {}).setdefault(n2, []).append(_new_callback)
                return
        callback()

    def _add_multerm(self, multerm):
        """Register a multerm so that we track matches against it"""
        # We only have to register each multerm once
        if multerm not in self._multerms_to_parents:
            self._multerms_to_parents[multerm] = []

            # For each child multerm add it recursively and set its parent and
            # indices of the term (h_idx) and of the multerm inside it (d_idx)
            for h_idx, h in enumerate(multerm.terms):
                for d_idx, d in enumerate(h.dst):
                    self._add_multerm(d)
                    self._multerms_to_parents[d].append((multerm, h_idx, d_idx))

                def _check(hyperedge, self=self, multerm=multerm, h_idx=h_idx):
                    self._check_new_hyperedge_multerm(hyperedge, multerm, h_idx)

                # When a hyperedge like this appears, check if it matches
                self._hyperedge_callbacks.setdefault((h.label, len(h.dst)), []).append(_check)

    def _check_new_hyperedge_multerm(self, hyperedge, multerm, h_idx):
        """Check if a new hyperedge matches h_idx term of multerm and if it does, propagate all
        the matyches up."""
        # The assuption that it is new is important: all matches will be relevant
        matches = self._get_hyperedge_matches(hyperedge, multerm.terms[h_idx])

        print("hyperedge", hyperedge, "try matching", multerm.terms[h_idx], "result", matches)

        if matches is None:
            return

        # Propagate the found matches up
        self._on_add_hyperedge_matches(multerm, h_idx, matches)

    def _get_hyperedge_matches(self, hyperedge, term):
        submatches = []
        for submulterm, d in zip(term.dst, hyperedge.dst):
            submatch = self._get_node_matches(d, submulterm)
            if submatch is not None:
                submatches.append(submatch)
            else:
                return None
        return HyperedgeMatches(hyperedge, submatches)

    def _get_node_matches(self, node, multerm):
        if not multerm.terms:
            # a trivial case where we don't require any outgoing hyperedge
            return NodeMatches(node, [])

        edgesetlist = self._node_to_multerms_to_edgesetlists.get(node, {}).get(multerm)
        if edgesetlist is not None and all(s for s in edgesetlist):
            return NodeMatches(node, [(self._get_hyperedge_matches(h, ph) for h in s)
                                      for ph, s in zip(multerm.terms, edgesetlist)])
        else:
            return None

    def _on_add_node_matches(self, multerm, matches):
        """We found new matches for a node. Propagate them up."""
        # First, call callbacks
        for cb in self._multerm_callbacks.get(multerm, []):
            cb(matches)
        # Check every parent
        for pmult, h_idx, d_idx in self._multerms_to_parents[multerm]:
            print("Checking parent", pmult, h_idx, d_idx)
            # Check if any incoming hyperedge may correspond to this term
            for h in matches.node.incoming:
                # We check the label, dst len and whether the d_idx dst is really this node
                if len(h.dst) == len(pmult.terms[h_idx].dst) and h.dst[d_idx] == matches.node and\
                        h.label == pmult.terms[h_idx].label:
                    # Now get the matches of this hyperedge against the term...
                    h_matches = self._get_hyperedge_matches(h, pmult.terms[h_idx])
                    if h_matches:
                        # ...and replace d_idx dst matches with our matches
                        h_matches.dst[d_idx] = matches
                        # The resulting macthes are the new matches which should be propagated
                        self._on_add_hyperedge_matches(pmult, h_idx, h_matches)

    def _on_add_hyperedge_matches(self, multerm, h_idx, matches):
        """We found new matches for the term h_idx of multerm, let's propagate this information"""
        hyperedge = matches.hyperedge

        # This is the list of sets of hyperedges of the src node.
        # Each set corresponds to a term and represents what hyperedges matches the term.
        src_edgesetlist = \
            self._node_to_multerms_to_edgesetlists\
            .setdefault(hyperedge.src, {})\
            .setdefault(multerm, [set() for s in multerm.terms])
        # Add the hyperedge for which we've found the matches to the right set
        src_edgesetlist[h_idx].add(hyperedge)

        # Construct new matches for the node given the new matches for the hyperedge
        # Basically, the matches for each term are taken from all the corresponding hyperedges.
        # Except for the h_idx term, for which we take only the new matches
        new_matches_terms = []
        for i, (ph, s) in enumerate(zip(multerm.terms, src_edgesetlist)):
            if i == h_idx:
                new_matches_terms.append([matches])
            else:
                if s:
                    new_matches_terms.append([self._get_hyperedge_matches(h, ph) for h in s])
                else:
                    # If there is a term without matches, the whole node is without matches
                    return

        print("Given matches", matches)
        print("we get new node matches", NodeMatches(hyperedge.src, new_matches_terms))

        # And propagate
        self._on_add_node_matches(multerm, NodeMatches(hyperedge.src, new_matches_terms))

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Hyperedge):
                for c in self._hyperedge_callbacks.get((e.label, len(e.dst)), []):
                    c(e)

    def on_merge(self, hypergraph, node, removed, added):
        new_callbacks = {}

        node_callbacks1 = self._merge_callbacks.pop(node, {})
        node_callbacks2 = self._merge_callbacks.pop(node.merged, {})
        for n, cbs in itertools.chain(node_callbacks1.items(), node_callbacks2.items()):
            n = n.follow()
            if n == node.merged:
                for cb in cbs:
                    cb()
            else:
                new_callbacks.setdefault(n, []).extend(cbs)

        self._merge_callbacks[node.merged] = new_callbacks

        mult_to_edgeslist = self._node_to_multerms_to_edgesetlists.pop(node, {})
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

def match_follow(match):
    return {k: v.follow() for k, v in match.items()}
