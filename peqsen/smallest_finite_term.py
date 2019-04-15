
import itertools
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, list_term_elements
import peqsen
import attr

@attr.s(slots=True, frozen=True)
class SelfSufficientNode:
    @property
    def merged(self):
        return self

#@peqsen.util.for_all_methods(peqsen.util.traced)
class SmallestHyperedgeTracker(Listener):
    def __init__(self, hypergraph=None, worst_value=float("inf"), measure=None,
                 auto_mark_self_sufficient=False):
        self.smallest = {}
        self.worst_value = worst_value
        if measure is None:
            measure = SmallestHyperedgeTracker.size
        self.measure = measure

        if hypergraph is not None:
            hypergraph.listeners.add(self)

        self.auto_mark_self_sufficient = auto_mark_self_sufficient

    @staticmethod
    def size(label, subvalues):
        return sum(subvalues) + 1

    @staticmethod
    def depth(label, subvalues):
        return max([0] + subvalues) + 1

    def mark_self_sufficient(self, node=None, value=1):
        if node is not None:
            node_smallest = self.smallest.setdefault(node, (self.worst_value, set()))
            if value < node_smallest[0]:
                self.smallest[node] = (value, {SelfSufficientNode()})
                self._update([node])
            elif value == node_smallest[0]:
                node_smallest[1].add(SelfSufficientNode())
        else:
            for n in self.smallest:
                if not n.outgoing:
                    self.mark_self_sufficient(n, value=value)

    def _update_node(self, node, to_update):
        for h in node.outgoing:
            self._update_hyperedge(h, to_update)
        for h in node.incoming:
            self._update_hyperedge(h, to_update)

    def _update_hyperedge(self, hyperedge, to_update):
        value = self.measure(hyperedge.label, [self.smallest.get(d, [self.worst_value])[0]
                                               for d in hyperedge.dst])
        src_smallest = self.smallest.setdefault(hyperedge.src, (self.worst_value, set()))
        if value < src_smallest[0]:
            self.smallest[hyperedge.src] = (value, {hyperedge})
            to_update.add(hyperedge.src)
        elif value == src_smallest[0]:
            src_smallest[1].add(hyperedge)
        elif hyperedge in src_smallest[1]:
            # tricky case: the size of the smallest hyperedge gets higher
            src_smallest[1].remove(hyperedge)
            if len(src_smallest[1]) == 0:
                self.smallest[hyperedge.src] = (self.worst_value, set())
                to_update.update(hyperedge.src.outgoing)

    def _update(self, elements):
        to_update = set(elements)
        while True:
            updating = list(to_update)
            to_update = set()
            for e in updating:
                if isinstance(e, Node):
                    self._update_node(e, to_update)
                else:
                    self._update_hyperedge(e, to_update)
            if len(to_update) == 0:
                return

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Node) and not e in self.smallest:
                self.smallest[e] = (self.worst_value, set())
        self._update(h for h in elements if isinstance(h, Hyperedge))

        if self.auto_mark_self_sufficient:
            for e in elements:
                if isinstance(e, Node) and not e.outgoing:
                    self.mark_self_sufficient(e)
        #  print()
        #  print("added", elements)
        #  print(hypergraph)
        #  print("smallest now =", self.smallest)

    def on_merge(self, hypergraph, node, removed, added, reason):
        # If some of the merged (removed) hyperedges were among the smallest,
        # we should update them in the corresponding sets (except for the outgoings of `node')
        for h in removed:
            if h.src != node:
                h_src_smallest_set = self.smallest[h.src][1]
                if h in h_src_smallest_set:
                    h_src_smallest_set.remove(h)
                    h_src_smallest_set.add(h.merged)

        # Now take care of the outgoings of node
        n_smallest = self.smallest[node]
        m_smallest = self.smallest[node.merged]
        if n_smallest[0] < m_smallest[0]:
            self.smallest[node.merged] = (n_smallest[0], set(h.merged for h in n_smallest[1]))
            self._update([node.merged])
        elif n_smallest[0] == m_smallest[0]:
            m_smallest[1].update(set(h.merged for h in n_smallest[1]))

        del self.smallest[node]

        self._update(added)
        #print()
        #print("merged", node)
        #print(hypergraph)
        #print("smallest now =", self.smallest)

    def _make_worst_rec(self, element, to_update):
        if isinstance(element, Hyperedge):
            if element.src in self.smallest:
                src_smallest = self.smallest[element.src]
                if element in src_smallest[1]:
                    if len(src_smallest[1]) > 1:
                        src_smallest[1].remove(element)
                    else:
                        self._make_worst_rec(element.src, to_update)
        elif isinstance(element, Node):
            self.smallest[element] = (self.worst_value, set())
            to_update.append(element)
            for e in element.incoming:
                self._make_worst_rec(e, to_update)

    def on_remove(self, hypergraph, elements):
        to_update = []
        for e in elements:
            self._make_worst_rec(e, to_update)
        self._update(to_update)
        #print()
        #print("removed", elements)
        #print(hypergraph)
        #print("smallest now =", self.smallest)

    def on_remove_node(self, hypergraph, node, hyperedges):
        del self.smallest[node]
        self.on_remove(hypergraph, hyperedges)
        #print()
        #print("removed node", node)
        #print(hypergraph)
        #print("smallest now =", self.smallest)

    def smallest_size(self, node):
        return self.smallest[node][0]

    def smallest_terms(self, node):
        for h in self.smallest[node][1]:
            if isinstance(h, SelfSufficientNode):
                yield node
            else:
                for subterms in itertools.product(*[self.smallest_terms(d) for d in h.dst]):
                    yield Term(h.label, subterms)

    def smallest_matches(self, node, top_node=None):
        for p in self._smallest_term_matches(node):
            yield {k: v for k, v in zip(list_term_elements(p[0], top_node=top_node), p[1])}

    def _smallest_term_matches(self, node):
        for h in self.smallest[node][1]:
            if isinstance(h, SelfSufficientNode):
                yield (node, [node])
            else:
                for subs in itertools.product(*[self._smallest_term_matches(d) for d in h.dst]):
                    yield (Term(h.label, [p[0] for p in subs]),
                           [h.src, h] + [e for p in subs for e in p[1]])

    def smallest_term_match_single(self, node, top_node=None):
        p = self._smallest_term_matches_single(node)
        return p[0], {k: v for k, v in zip(list_term_elements(p[0], top_node=top_node), p[1])}

    def _smallest_term_matches_single(self, node):
        h = next(iter(self.smallest[node][1]))
        if isinstance(h, SelfSufficientNode):
            return (node, [node])
        else:
            subs = [self._smallest_term_matches_single(d) for d in h.dst]
            return (Term(h.label, [p[0] for p in subs]),
                    [h.src, h] + [e for p in subs for e in p[1]])


def measure_term(term, measure):
    return measure(term.label, [measure_term(d, measure) for d in term.dst])

def finite_terms(node, max_depth=float('inf')):
    def _finite_terms(node, history, max_depth=max_depth):
        if len(history) >= max_depth or node in history:
            return
        else:
            new_history = history | {node}
            for h in node.outgoing:
                for subterms in itertools.product(*[_finite_terms(d, new_history)
                                                    for d in h.dst]):
                    yield Term(h.label, subterms)

    return _finite_terms(node, set(), max_depth)
