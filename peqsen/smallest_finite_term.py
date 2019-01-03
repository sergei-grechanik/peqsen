
import itertools
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term
import peqsen

# TODO: This contains mistakes and should be rewritten
@peqsen.util.for_all_methods(peqsen.util.traced)
class SmallestHyperedgeTracker(Listener):
    def __init__(self, worst_value=float("inf"), measure=None):
        self.smallest = {}
        self.worst_value = worst_value
        if measure is None:
            measure = SmallestHyperedgeTracker.size
        self.measure = measure

    @staticmethod
    def size(label, subvalues):
        return sum(subvalues) + 1

    @staticmethod
    def depth(label, subvalues):
        return max([0] + subvalues) + 1

    def _update_node(self, node, to_update):
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
        peqsen.util.printind("smallest now =", self.smallest)

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
        peqsen.util.printind("smallest now =", self.smallest)

    def _make_worst_rec(self, element):
        if isinstance(element, Hyperedge):
            if element.src in self.smallest:
                src_smallest = self.smallest[element.src]
                if element in src_smallest[1]:
                    if len(src_smallest[1]) > 1:
                        src_smallest[1].remove(element)
                    else:
                        self._make_worst_rec(element.src)
                        return True
            return False
        elif isinstance(element, Node):
            self.smallest[element] = (self.worst_value, set())
            for e in element.incoming:
                self._make_worst_rec(e)
            return True

    def on_remove(self, hypergraph, elements):
        to_update = []
        for e in elements:
            if self._make_worst_rec(e):
                to_update.extend(list(e.src.outgoing) + [e.src])
        self._update(to_update)
        peqsen.util.printind("smallest now =", self.smallest)

    def on_remove_node(self, hypergraph, node, hyperedges):
        del self.smallest[node]
        self.on_remove(hypergraph, hyperedges)
        peqsen.util.printind("smallest now =", self.smallest)

    def smallest_terms(self, node):
        for h in self.smallest[node][1]:
            for subterms in itertools.product(*[self.smallest_terms(d) for d in h.dst]):
                yield Term(h.label, subterms)


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
