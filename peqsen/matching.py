
import itertools
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term

class TriggerManager(Listener):
    def __init__(self):
        self._node_multerms_to_matches = {}
        self._multerm_callbacks = {}
        self._merge_callbacks = {}
        self._multerms = set()
        self._hyperedge_callbacks = {}

    def add_trigger(self, pattern, callback):
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

        def _on_this_multerm(matched, pat_to_index=pat_to_index, index_mergelist=index_mergelist,
                             self=self, callback=callback):
            need_merged = [(matched[p[0]], matched[p[1]]) for p in index_mergelist]

            def _on_pattern_matched(matched=matched, pat_to_index=pat_to_index
                                    self=self, callback=callback):
                match = {n: matched[i].follow() for n, i in pat_to_index.items()}
                if all(e in self.hypergraph for e in match.values()):
                    callback(match)

            self._add_multimerge_callback(need_merged, _on_pattern_matched)

        self._multerm_callbacks.setdefault(multerm, []).append(_on_this_multerm)

    def _pattern_to_multerm(self, pattern):
        if isinstance(pattern, Node):
            matchlist = [pattern]
            subterms = []
            for h in pattern.outgoing:
                mterm, mlst = self._pattern_to_multerm(h)
                matchlist.extend(mlst)
                subterms.append(mterm)
            return tuple(sorted(subterms)), matchlist
        elif isinstance(pattern, Hyperedge):
            matchlist = []
            subterms = []
            for d in pattern.dst:
                mterm, mlst = self._pattern_to_multerm(d)
                matchlist.extend(mlst)
                subterms.append(mterm)
            return Term(pattern.label, subterms), matchlist

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
        if multerm not in self._multerms:
            self._multerms.add(multerm)
            for idx, h in enumerate(multerm):
                for d in h.dst:
                    self._add_multerm(d)

                def _check(hyperedge, self=self, multerm=multerm, idx=idx):
                    self._check_hyperedge_multerm(hyperedge, multerm, idx)

                self._hyperedge_callbacks.setdefault((h.label, len(h.dst)), []).append(_check)

    def _check_hyperedge_multerm(self, hyperedge, multerm, idx):
        submatches = []
        for submulterm, d in zip(multerm[idx].dst, hyperedge.dst):
            d_matches = self._node_multerms_to_matches.get((d, submulterm))
            if d_matches:
                submatches.append(d_matches)
            else:
                return
        for sub in itertools.product(*submatches):
            

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Hyperedge):
                callbacks = self._hyperedge_callbacks.get((e.label, len(e.dst)))
                if callbacks:
                    for c in callbacks:
                        c(e)

    def on_merge(self, hypergraph, node, removed, added):
        pair = tuple(sorted((node, node.merged)))
        for cb in self._merge_callbacks[pair]:
            cb()
        del self._merge_callbacks[pair]

        

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
