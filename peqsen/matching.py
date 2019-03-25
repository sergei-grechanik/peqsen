
import itertools
import collections
import attr

import peqsen.util
# TODO: Remove all printinds
#from peqsen.util import # printind
# printind = lambda *_: None
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, list_term_elements

@attr.s(slots=True, frozen=True)
class Multerm:
    """Used internally in TriggerManager. Multerm is a list of terms that can be matched against a
    node, meaning that the node must have hyperedges matching these terms."""
    terms = attr.ib()

@attr.s(slots=True, frozen=True)
class NodeMatches:
    """Used internally in TriggerManager. NodeMatches is a (set of) matching of a node against a
    Multerm, terms must be a list of lists of HyperedgeMatches (that is, each term may match several
    hyperedges, so this represents a collection of matches)"""
    node = attr.ib()
    terms = attr.ib()

@attr.s(slots=True, frozen=True)
class HyperedgeMatches:
    """Used internally in TriggerManager. HyperedgeMatches is a matching of a hyperedge against a
    Term, dst must be a list of NodeMatches"""
    hyperedge = attr.ib()
    dst = attr.ib()

#@peqsen.util.for_all_methods(peqsen.util.traced)
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
        # [(Node) -> ()]
        self._node_callbacks = []

    def add_trigger(self, pattern, callback):
        if isinstance(pattern, Term):
            pattern = list_term_elements(pattern)[0]

        assert isinstance(pattern, Node)

        if len(pattern.outgoing) == 0:
            # special case
            def _on_new_node(node, pattern=pattern, callback=callback):
                callback({pattern: node})
            self._node_callbacks.append(_on_new_node)
            return

        multerm, matchlist = self._pattern_to_multerm(pattern)

        # printind("\n================")
        # printind(pattern)
        # printind(multerm)
        # printind(matchlist)
        # printind("================\n")

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
            # printind()
            # printind("Multerm", multerm)
            # printind("Matched", matches)
            for matched in self._matches_to_matchlists(matches):
                # printind("    list", matched)
                # printind("matchlist", matchlist)
                need_merged = [(matched[p[0]], matched[p[1]]) for p in index_mergelist]
                # printind("need merged", need_merged)

                def _on_pattern_matched(matched=matched, pat_to_index=pat_to_index,
                                        self=self, callback=callback):
                    match = {n: matched[i].follow() for n, i in pat_to_index.items()}
                    if all(e in self.hypergraph for e in match.values()):
                        callback(match)

                self._add_multimerge_callback(need_merged, _on_pattern_matched)

        self._multerm_callbacks.setdefault(multerm, []).append(_on_this_multerm)

        # Since existing nodes may match the multerm, we have to trigger the new callback
        for n in self.hypergraph.nodes():
            existing_matches = self._get_node_matches(n, multerm)
            if existing_matches:
                _on_this_multerm(existing_matches)

        # We do this at the end because this will immediately trigger stuff for existing hyperedges
        self._add_multerm(multerm)

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
            return Term(pattern.label, tuple(subterms)), matchlist

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
            n1 = n1.follow()
            n2 = n2.follow()
            if n1 != n2:
                if n1 > n2:
                    n1, n2 = n2, n1
                def _new_callback(self=self, need_merged=need_merged, callback=callback):
                    # printind("Still need merged:", need_merged)
                    self._add_multimerge_callback(need_merged, callback)
                self._merge_callbacks.setdefault(n1, {}).setdefault(n2, []).append(_new_callback)
                return
        callback()

    def _add_multerm(self, multerm):
        """Register a multerm so that we track matches against it"""
        # this is a dict of new callbacks, which will be used locally on existing hyperedges
        new_callbacks = {}

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

                # add the same callback to the list of new ones
                new_callbacks.setdefault((h.label, len(h.dst)), []).append(_check)

        # run callbacks for existing hyperedges
        for h in self.hypergraph.hyperedges():
            for c in new_callbacks.get((h.label, len(h.dst)), []):
                c(h)

    def _check_new_hyperedge_multerm(self, hyperedge, multerm, h_idx):
        """Check if a new hyperedge matches h_idx term of multerm and if it does, propagate all
        the matyches up."""
        # The assuption that it is new is important: all matches will be relevant
        matches = self._get_hyperedge_matches(hyperedge, multerm.terms[h_idx])

        # printind("hyperedge", hyperedge, "try matching", multerm.terms[h_idx], "result", matches)

        if matches is None:
            return

        # Propagate the found matches up
        self._on_add_hyperedge_matches(multerm, h_idx, matches)

    def _get_hyperedge_matches(self, hyperedge, term, check_non_null=False):
        hyperedge = hyperedge.follow()
        submatches = []
        for submulterm, d in zip(term.dst, hyperedge.dst):
            submatch = self._get_node_matches(d, submulterm, check_non_null)
            if submatch is not None:
                submatches.append(submatch)
            else:
                return None
        return HyperedgeMatches(hyperedge, submatches)

    def _get_node_matches(self, node, multerm, check_non_null=False):
        # printind("_get_node_matches", node, multerm)
        if not multerm.terms:
            # a trivial case where we don't require any outgoing hyperedge
            return NodeMatches(node, [])

        edgesetlist = self._node_to_multerms_to_edgesetlists.get(node, {}).get(multerm)
        # printind("edgesetlist", edgesetlist)
        if edgesetlist is not None and all(s for s in edgesetlist):
            return NodeMatches(node, [[self._get_hyperedge_matches(h, ph, True) for h in s]
                                      for ph, s in zip(multerm.terms, edgesetlist)])
        elif check_non_null:
            raise AssertionError("No matches for node {} and multerm {}".format(node, multerm))
        else:
            # printind("result none")
            return None

    def _on_add_node_matches(self, multerm, matches):
        """We found new matches for a node. Propagate them up."""
        assert len(multerm.terms) == len(matches.terms)
        # First, call callbacks
        for cb in self._multerm_callbacks.get(multerm, []):
            cb(matches)
        # Check every parent
        for pmult, h_idx, d_idx in self._multerms_to_parents[multerm]:
            # printind("Checking parent", pmult, h_idx, d_idx)
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

        assert len(multerm.terms[h_idx].dst) == len(hyperedge.dst)
        assert len(matches.dst) == len(hyperedge.dst)

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
                    new_matches_terms.append([self._get_hyperedge_matches(h, ph, True) for h in s])
                else:
                    # If there is a term without matches, the whole node is without matches
                    return

        # printind("Given matches", matches)
        # printind("we get new node matches", NodeMatches(hyperedge.src, new_matches_terms))

        # And propagate
        self._on_add_node_matches(multerm, NodeMatches(hyperedge.src, new_matches_terms))

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Hyperedge):
                for c in self._hyperedge_callbacks.get((e.label, len(e.dst)), []):
                    c(e)
            elif isinstance(e, Node):
                for c in self._node_callbacks:
                    c(e)

    def on_merge(self, hypergraph, node, removed, added, reason):
        # printind("On merge", node)
        # printind("merge cb: ", self._merge_callbacks)

        # Pop callback maps for both nodes
        node_callbacks1 = self._merge_callbacks.pop(node, {})
        node_callbacks2 = self._merge_callbacks.pop(node.merged, {})
        # And create a new one for the node we are merging into
        new_main_node_callbacks = self._merge_callbacks.setdefault(node.merged, {})
        # Iterate through callbacks from both nodes
        for n, cbs in itertools.chain(node_callbacks1.items(), node_callbacks2.items()):
            n = n.follow()
            if n == node.merged:
                # This callback fires
                for cb in cbs:
                    cb()
                # Note that callbacks may add new callbacks, so new_main_node_callbacks might be
                # different now
            else:
                # This callback doesn't fire, readd it back
                new_main_node_callbacks.setdefault(n, []).extend(cbs)

        # printind("new merge cb: ", self._merge_callbacks)

        new_matches = []

        mult_to_edgeslist = self._node_to_multerms_to_edgesetlists.pop(node, {})
        # printind("Node", node, "mult_to_edgelist", mult_to_edgeslist)
        for multerm, edgesetlist in mult_to_edgeslist.items():
            new_edgesetlist = \
                self._node_to_multerms_to_edgesetlists\
                .setdefault(node.merged, {})\
                .setdefault(multerm, [set() for s in multerm.terms])
            # printind("new_edgesetlist", new_edgesetlist)
            was_unfull = False
            became_full = True
            for i in range(len(edgesetlist)):
                if not new_edgesetlist[i]:
                    was_unfull = True
                    if not edgesetlist[i]:
                        became_full = False
                new_edgesetlist[i].update(h.follow() for h in edgesetlist[i])
            # printind("new_edgesetlist", new_edgesetlist)
            if was_unfull and became_full:
                # printind("was unfull, became full")
                new_matches.append((multerm, new_edgesetlist))

        # Note that we propagate this after changing the edgelists because
        # _get_hyperedge_matches may use these edgesetlist in case of recursive stuff
        for multerm, new_edgesetlist in new_matches:
            node_matches = NodeMatches(node.merged,
                                       [[self._get_hyperedge_matches(h, ph, True) for h in s]
                                        for ph, s in zip(multerm.terms, new_edgesetlist)])
            # printind(node_matches)
            self._on_add_node_matches(multerm, node_matches)

        # Merged incoming hyperedges must be checked
        self.on_add(hypergraph, (h for h in added if node.merged in h.dst))

    def on_remove(self, hypergraph, elements):
        to_remove_node_matches = []
        for h in elements:
            multerms_to_edgesetlists = self._node_to_multerms_to_edgesetlists.get(h.src)
            if multerms_to_edgesetlists:
                for multerm, edgesetlist in multerms_to_edgesetlists.items():
                    was_full = True
                    became_unfull = False
                    for term, edgeset in zip(multerm.terms, edgesetlist):
                        if edgeset:
                            new_edgeset = [e for e in (e.follow() for e in edgeset) if e != h]
                            edgeset.clear()
                            edgeset.update(new_edgeset)
                            if not edgeset:
                                became_unfull = True
                        else:
                            was_full = False
                    if was_full and became_unfull:
                        to_remove_node_matches.append((h.src, multerm))

        for node, multerm in to_remove_node_matches:
            self._on_remove_node_matches(node, multerm)

    def _on_remove_node_matches(self, node, multerm):
        # Note that here all matches of the multerm are removed, not some matches, and this is
        # the difference with adding matches
        for pmult, h_idx, d_idx in self._multerms_to_parents[multerm]:
            # printind("Checking parent", pmult, h_idx, d_idx)
            # Same as in add, check if any incoming hyperedge may correspond to this term
            for h in node.incoming:
                # We check the label, dst len and whether the d_idx dst is really this node
                if len(h.dst) == len(pmult.terms[h_idx].dst) and h.dst[d_idx] == node and\
                        h.label == pmult.terms[h_idx].label:
                    # Now this hyperedge can't match the term
                    self._on_remove_hyperedge_matches(h, pmult, h_idx)

    def _on_remove_hyperedge_matches(self, hyperedge, multerm, h_idx):
        multerms_to_edgesetlists = self._node_to_multerms_to_edgesetlists.get(hyperedge.src)
        if multerms_to_edgesetlists:
            edgesetlist = multerms_to_edgesetlists.get(multerm)
            if edgesetlist:
                edgeset = edgesetlist[h_idx]
                if edgeset:
                    new_edgeset = [e for e in (e.follow() for e in edgeset) if e != hyperedge]
                    edgeset.clear()
                    edgeset.update(new_edgeset)
                    if not edgeset:
                        self._on_remove_node_matches(hyperedge.src, multerm)

    def on_remove_node(self, hypergraph, node, hyperedges):
        self._node_to_multerms_to_edgesetlists.pop(node, None)
        self.on_remove(hypergraph, [h for h in hyperedges if h.src != node])

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

def still_match(match, hypergraph):
    """Check if all the hyperedges and nodes participating in the match are still present in
    the hypergraph. Note that it doesn't check if it is really a match."""
    for v in match.values():
        if v not in hypergraph:
            return False
    return True

