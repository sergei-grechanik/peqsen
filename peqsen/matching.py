
import itertools
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term

class TriggerManager(Listener):
    def add_trigger(pattern, callback):
        pass
    def on_add(self, hypergraph, elements):
        pass
    def on_merge(self, hypergraph, node, removed, added):
        pass
    def on_remove(self, hypergraph, elements):
        pass
    def on_remove_node(self, hypergraph, node, hyperedges):
        pass

def combine_submatches(*args):
    res = {}
    for m in args:
        for k, v in m.items():
            if res.setdefault(k, v) != v:
                return None
    return res

def find_matches_hyperedge(hyperedge, pattern):
    if hyperedge.label == pattern.label and len(hyperedge.dst) == len(pattern.dst):
        submatches = [find_matches_node(d, p) for d, p in zip(hyperedge.dst, pattern.dst)]
        for sub in itertools.product(*submatches):
            combined = combine_submatches(*sub)
            if combined is not None:
                combined[pattern] = hyperedge
                yield combined

def find_matches_node(node, pattern):
    submatches = [itertools.chain(*(find_matches_hyperedge(h, p) for h in node.outgoing))
                  for p in pattern.outgoing]
    for sub in itertools.product(*submatches):
        combined = combine_submatches(*sub)
        if combined is not None:
            combined[pattern] = node
            yield combined

def find_matches(hypergraph, pattern):
    for n in hypergraph.nodes():
        for m in find_matches_node(n, pattern):
            yield m
