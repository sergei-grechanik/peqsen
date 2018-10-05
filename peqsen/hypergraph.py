
import itertools
import networkx
import hypothesis
from hypothesis import strategies

def some_name(obj, length=5):
    h = hash(obj)
    res = ""
    consonant = True
    consonants = "rtpsdfghjklzxvbnm"
    vowels = "euioa"
    for i in range(length):
        if consonant:
            res += consonants[h % len(consonants)]
            h //= len(consonants)
        else:
            res += vowels[h % len(vowels)]
            h //= len(vowels)
        consonant = not consonant
    return res

class GloballyIndexed:
    global_index = 0

    @staticmethod
    def reset_global_index():
        GloballyIndexed.global_index = 0

    def __init__(self):
        GloballyIndexed.global_index += 1
        self._global_index = GloballyIndexed.global_index

    def __hash__(self):
        return hash(self._global_index)

class Node(GloballyIndexed):
    def __init__(self):
        super().__init__()
        self.incoming = set()
        self.outgoing = set()
        self.to_be_removed = False
        self.merged = None

    def __repr__(self):
        return some_name(hash(self)) + str(self._global_index) + \
            ("(=" + repr(self.merged) + ")" if self.merged else "")

    def follow(self):
        return self if self.merged is None else self.merged.follow()

    def apply_map(self, mapping):
        res = mapping.get(self)
        return self.follow() if res is None else res.follow()


class Hyperedge(GloballyIndexed):
    def __init__(self, label, src=None, dst=None):
        super().__init__()

        if dst is None:
            dst = []

        self.src = src
        self.dst = dst
        self.label = label
        self.to_be_removed = False
        self.merged = None

    def __repr__(self):
        return repr(self.src) + "->" + repr(self.label) + "{" + some_name(hash(self), 3) + "}" + \
            "->" + repr(self.dst) + \
            ("(=" + repr(self.merged) + ")" if self.merged else "")

    def follow(self):
        if self.merged is None:
            if (self.src is None or self.src.merged is None) and \
                    all(d.merged is None for d in self.dst):
                return self
            else:
                return Hyperedge(self.label, None if self.src is None else self.src.follow(),
                                 [d.follow() for d in self.dst])
        else:
            return self.merged.follow()

    def apply_map(self, mapping):
        res = mapping.get(self)
        if res is None:
            return Hyperedge(self.label, None if self.src is None else self.src.apply_map(mapping),
                             [d.apply_map(mapping) for d in self.dst])
        else:
            return res.follow()


class Recursively:
    def __init__(self, node):
        self.node = node

    def follow(self):
        return Recursively(self.node.follow())

class Term:
    def __init__(self, label, dst=None):
        if dst is None and isinstance(label, tuple):
            dst = [s if isinstance(s, (Node, Term)) else Term(s) for s in label[1:]]
            label = label[0]

        if dst is None:
            dst = []

        self.hyperedge = Hyperedge(label, None, dst)

    @property
    def outgoing(self):
        return [self.hyperedge]

    @property
    def label(self):
        return self.hyperedge.label

    @property
    def dst(self):
        return self.hyperedge.dst

    def follow(self):
        return self

    def __repr__(self):
        return repr(self.hyperedge.label) + repr(self.hyperedge.dst)

    def __eq__(self, other):
        return self.label == other.label and self.dst == other.dst

    def __hash__(self):
        return hash((self.label, tuple(self.dst)))


class Listener:
    def on_add(self, hypergraph, elements):
        pass
    def on_merge(self, hypergraph, node, removed, added):
        pass
    def on_remove(self, hypergraph, elements):
        pass
    def on_remove_node(self, hypergraph, node, hyperedges):
        pass

class Hypergraph:
    def __init__(self):
        self._being_modified = False
        self._nodes = set()
        self._hyperedges = {}
        self.listeners = set()

    def __repr__(self):
        res = object.__repr__(self) + " {\n"
        for n in self._nodes:
            res += "    " + str(n) + " {\n"
            for h in n.outgoing:
                mark = '' if self._hyperedges.get((h.label, tuple(h.dst))) is h else '! '
                res += "        " + mark + str(h) + "\n"
            res += "    }\n"
        res += "}"
        return res

    def nodes(self):
        return self._nodes

    def hyperedges(self):
        return self._hyperedges.values()

    def __contains__(self, element):
        if isinstance(element, Node):
            return element in self._nodes
        elif isinstance(element, Hyperedge):
            return element.src in self._nodes and element in element.src.outgoing
        else:
            raise ValueError("Cannot check whether this is in hypergraph: {}".format(element))

    def on_add(self, elements):
        for l in self.listeners:
            l.on_add(self, elements)

    def on_merge(self, node, removed, added):
        for l in self.listeners:
            l.on_merge(self, node, removed, added)

    def on_remove(self, elements):
        for l in self.listeners:
            l.on_remove(self, elements)

    def on_remove_node(self, node, hyperedges):
        for l in self.listeners:
            l.on_remove_node(self, node, hyperedges)

    def add_from(self, hypergraph):
        to_add = \
            sorted(hypergraph.hyperedges(), key=lambda h: len(h.dst)) + list(hypergraph.nodes())
        added = self.rewrite(add=to_add)
        return dict(zip(to_add, added))

    def rewrite(self, remove=(), add=(), merge=(), ignore_already_removed=False):
        if self._being_modified:
            raise RuntimeError("Trying to modify a hypergraph during another modification")

        if isinstance(remove, (Hyperedge, Node)):
            remove = [remove]

        if isinstance(add, (Node, Hyperedge, Term)):
            add = [add]

        remove = [e.follow() for e in remove]
        add = [e.follow() for e in add]
        merge = [[e.follow() for e in es] for es in merge]

        self._being_modified = True

        self._remove(remove, 0, ignore_already_removed=ignore_already_removed)

        added = []
        to_merge = list(merge)
        resmap = {}
        add_res = self._add(add, added, to_merge, resmap)
        self.on_add(added)

        self._merge(to_merge)

        removed = []
        self._remove((h.follow() for h in remove), 1, removed,
                     ignore_already_removed=ignore_already_removed)
        self.on_remove(removed)

        self._being_modified = False

        return [e.follow() for e in add_res]

    def remove_nodes(self, nodes):
        if self._being_modified:
            raise RuntimeError("Trying to modify a hypergraph during another modification")

        if isinstance(nodes, Node):
            nodes = [nodes]

        for n in nodes:
            if n in self._nodes:
                self._being_modified = True

                removed = []
                for h in list(itertools.chain(n.incoming, n.outgoing)):
                    if h in h.src.outgoing:
                        self._remove_hyperedge(h, 0, [])
                        self._remove_hyperedge(h, 1, removed)
                self._nodes.remove(n)
                self.on_remove_node(n, removed)

                self._being_modified = False
            else:
                raise ValueError("Cannot remove a node which is not in the hypergraph: {}"
                                 .format(n))

    def _remove(self, elements, phase, removed=None, ignore_already_removed=False):
        for e in elements:
            if isinstance(e, Node):
                raise ValueError("Manual removal of nodes is forbidden")
            elif isinstance(e, Hyperedge):
                # Check if hyperedge exists only on 0th phase
                if phase == 0:
                    if e.src is None or e.src not in self._nodes or e not in e.src.outgoing:
                        if ignore_already_removed:
                            continue
                        else:
                            raise ValueError("Hyperedge cannot be removed: {}", e)
                self._remove_hyperedge(e, phase, removed)
            else:
                raise ValueError("Cannot remove a thing of unknown type: {}".format(e))

    def _remove_hyperedge(self, hyperedge, phase, removed):
        if phase == 0:
            hyperedge.to_be_removed = True
        elif phase == 1:
            if hyperedge.to_be_removed:
                removed.append(hyperedge)
                hyperedge.to_be_removed = False
                hyperedge.src.outgoing.remove(hyperedge)
                for dnode in hyperedge.dst:
                    dnode.incoming.discard(hyperedge)
                if self._hyperedges.get((hyperedge.label, tuple(hyperedge.dst))) is hyperedge:
                    del self._hyperedges[(hyperedge.label, tuple(hyperedge.dst))]

    def _merge(self, merge):
        if not merge:
            return
        to_merge = []
        for n1, n2 in merge:
            n1 = n1.follow()
            n2 = n2.follow()
            if n1 != n2:
                # Sorting may break our implementation of cong closure, not sure
                # if len(n1.incoming) + len(n1.outgoing) > len(n2.incoming) + len(n2.outgoing):
                #     (n1, n2) = (n2, n1)
                n1.merged = n2
                removed = []
                added = []
                # This order: out then in is important
                for h in list(itertools.chain(n1.outgoing, n1.incoming)):
                    if h.merged is None:
                        new_src = n2 if h.src == n1 else h.src
                        new_dst = [n2 if n == n1 else n for n in h.dst]

                        if self._hyperedges.get((h.label, tuple(h.dst))) is h:
                            # bad case: merging of a hyperedge which is the main one
                            del self._hyperedges[(h.label, tuple(h.dst))]

                        if h.merged is None:
                            new_h = Hyperedge(h.label, new_src, new_dst)
                            new_h.to_be_removed = h.to_be_removed
                            h.merged = self._add_hyperedge(new_h, added, to_merge)
                        h.to_be_removed = True
                        self._remove_hyperedge(h, 1, removed)
                self._nodes.remove(n1)
                self.on_merge(n1, removed, added)
        self._merge(to_merge)

    def _add(self, elements, added, to_merge, resmap):
        res = []
        for e in elements:
            recursively = False
            if isinstance(e, Recursively):
                recursively = True
                e = e.node

            if isinstance(e, Node):
                if e in self._nodes:
                    res.append(e)
                elif e in resmap:
                    n = resmap[e]
                    if n is None:
                        n = Node()
                        self._nodes.add(n)
                        added.append(n)
                        resmap[e] = n
                    res.append(resmap[e])
                elif recursively:
                    resmap[e] = None
                    for h in e.outgoing:
                        if h.src is not None and h.src != e:
                            raise ValueError("The source of a hyperedge being added is not the same"
                                             "as its parent node")
                        new_dst = self._add([Recursively(d) for d in h.dst],
                                            added, to_merge, resmap)
                        h_res = self._add_hyperedge(Hyperedge(h.label, resmap[e], new_dst),
                                                    added, to_merge)
                        resmap[e] = h_res.src
                        resmap[h] = h_res
                    res.append(resmap[e])
                else:
                    n = Node()
                    self._nodes.add(n)
                    resmap[e] = n
                    added.append(n)
                    res.append(n)
            elif isinstance(e, (Hyperedge, Term)):
                if isinstance(e, Term):
                    e = e.hyperedge
                if e in resmap:
                    res.append(resmap[e])
                else:
                    if recursively:
                        new_dst = self._add([Recursively(d) for d in e.dst],
                                            added, to_merge, resmap)
                    else:
                        new_dst = self._add(e.dst, added, to_merge, resmap)

                    if e.src is None or e.src in self._nodes:
                        h = self._add_hyperedge(Hyperedge(e.label, e.src, new_dst),
                                                added, to_merge)
                    elif e.src in resmap:
                        h = self._add_hyperedge(Hyperedge(e.label, resmap[e.src], new_dst),
                                                added, to_merge)
                    else:
                        h = self._add_hyperedge(Hyperedge(e.label, None, new_dst),
                                                added, to_merge)
                        resmap[e.src] = h.src

                    resmap[e] = h
                    res.append(h)
            else:
                raise ValueError("Cannot add this, unknown type: {}".format(e))
        return res

    def _add_hyperedge(self, hyperedge, added, to_merge):
        # First, check if an equivalent hyperedge exists
        existing = self._hyperedges.get((hyperedge.label, tuple(hyperedge.dst)))
        if existing is not None:
            if hyperedge.src == existing.src or hyperedge.src is None:
                # Good, nothing to do, just mark that we don't want to remove it
                existing.to_be_removed = existing.to_be_removed and hyperedge.to_be_removed
                return existing
            else:
                # in this case we have to merge the sources, and continue adding the hyperedge
                to_merge.append((hyperedge.src, existing.src))

        # In some cases there may be an existing hyperedge which wasn't registered in _hyperedges
        if hyperedge.src is not None:
            for h in hyperedge.src.outgoing:
                if h.label == hyperedge.label and h.dst == hyperedge.dst:
                    h.to_be_removed = h.to_be_removed and hyperedge.to_be_removed
                    if existing is None:
                        self._hyperedges[(hyperedge.label, tuple(hyperedge.dst))] = h
                    return h

        # Now this hyperedge should be added as a new hyperedge

        # Crete the source
        if hyperedge.src is None:
            hyperedge.src = Node()
            self._nodes.add(hyperedge.src)
            added.append(hyperedge.src)

        hyperedge.src.outgoing.add(hyperedge)
        for d in hyperedge.dst:
            d.incoming.add(hyperedge)

        added.append(hyperedge)

        # Register the hyperedge if it wasn't registered before
        if existing is None:
            self._hyperedges[(hyperedge.label, tuple(hyperedge.dst))] = hyperedge

        return hyperedge

    def check_integrity(self, strict=True):
        """Sometimes hypergraph may be in a non-congruent-closed state. It may happen between
        on_add and on_merge when a node gets an outgoing hyperedge which leads to its merging, but
        it cannot be merged yet. In this case use `strict=False`."""
        for n in self._nodes:
            assert n.merged is None
            for h in n.outgoing:
                assert h.merged is None
                assert h.src == n
                assert all(d in self._nodes for d in h.dst)
                assert all(h in d.incoming for d in h.dst)
                if strict:
                    assert not h.to_be_removed
                    assert self._hyperedges[(h.label, tuple(h.dst))] is h
            for h in n.incoming:
                assert h.merged is None
                assert h.src in self._nodes
                assert all(d in self._nodes for d in h.dst)
                assert all(h in d.incoming for d in h.dst)
                if strict:
                    assert not h.to_be_removed
                    assert self._hyperedges[(h.label, tuple(h.dst))] is h
        for h in self.hyperedges():
            assert h.src in self._nodes

    def as_networkx(self):
        graph = networkx.MultiDiGraph()
        graph.add_nodes_from(id(n) for n in self._nodes)
        graph.add_nodes_from(id(h) for h in self._hyperedges.values())
        for h in self._hyperedges.values():
            graph.add_edge(id(h.src), id(h))
            graph.add_edges_from((id(h), id(d), i) for i, d in enumerate(h.dst))
        return graph

    def isomorphic(self, other):
        return networkx.algorithms.isomorphism.is_isomorphic(self.as_networkx(),
                                                             other.as_networkx())

def map_rewrite(rewrite, mapping):
    return {'remove': [mapping[h] for h in rewrite['remove']],
            'add': [x.apply_map(mapping) for x in rewrite['add']],
            'merge': [tuple(mapping[x] for x in p) for p in rewrite['merge']]}

DEFAULT_MAX_CHILDREN=4
DEFAULT_LABELS=('a', 'b', 'c')

@strategies.composite
def gen_hyperedge(draw, nodes, labels=DEFAULT_LABELS, max_children=DEFAULT_MAX_CHILDREN):
    if isinstance(labels, (list, tuple)):
        labels = strategies.sampled_from(labels)
    if isinstance(nodes, (list, tuple)):
        nodes = strategies.sampled_from(nodes)
    src = draw(nodes)
    dst = draw(strategies.lists(nodes, max_size=max_children))
    label = draw(labels)
    return Hyperedge(label, src, dst)

@strategies.composite
def gen_simple_addition(draw, labels=DEFAULT_LABELS,
                        max_nodes=10, max_hyperedges=100, max_children=DEFAULT_MAX_CHILDREN):
    nodes = [Node() for i in range(draw(strategies.integers(0, max_nodes)))]
    hyperedges = draw(strategies.lists(gen_hyperedge(nodes, labels, max_children),
                                       max_size=max_hyperedges))
    if draw(strategies.booleans()):
        size = len(nodes) + len(hyperedges)
        return draw(strategies.lists(strategies.sampled_from(nodes + hyperedges),
                                     max_size=size, min_size=size))
    else:
        return draw(strategies.permutations(nodes + hyperedges))

@strategies.composite
def gen_rewrite(draw, hypergraph, labels=DEFAULT_LABELS,
                max_add_nodes=10, max_add_hyperedges=20, max_children=DEFAULT_MAX_CHILDREN,
                max_remove=20, max_merge=20):
    add_nodes = [Node() for i in range(draw(strategies.integers(0, max_add_nodes)))]
    nodes = list(hypergraph.nodes())
    add_hyperedges = draw(strategies.lists(gen_hyperedge(add_nodes + nodes,
                                                         labels=labels, max_children=max_children),
                                           max_size=max_add_hyperedges))
    hyperedges = list(hypergraph.hyperedges())
    remove = draw(strategies.lists(strategies.sampled_from(hyperedges), max_size=max_remove))
    merge = draw(strategies.lists(strategies.tuples(strategies.sampled_from(nodes),
                                                    strategies.sampled_from(nodes)),
                                  max_size=max_merge))
    add = draw(strategies.permutations(add_nodes + add_hyperedges))
    return {'remove': remove, 'add': add, 'merge': merge}

@strategies.composite
def gen_permuted_rewrite(draw, rewrite):
    res = {}
    for key in rewrite:
        res[key] = draw(strategies.permutations(rewrite[key]))
    return res

@strategies.composite
def gen_hypergraph(draw, labels=strategies.sampled_from(['a', 'b']),
                   max_nodes=10, max_hyperedges=100, max_children=DEFAULT_MAX_CHILDREN):
    to_add = draw(gen_simple_addition(labels=labels,
                                      max_nodes=max_nodes,
                                      max_hyperedges=max_hyperedges,
                                      max_children=max_children))
    h = Hypergraph()
    h.rewrite(add=to_add)
    return h


# TODO: Move this class somewhere
class BestHyperedgeTracker(Listener):
    def __init__(self, worst_value=float("inf"), measure=None):
        self.best = {}
        self.worst_value = worst_value
        if measure is None:
            measure = BestHyperedgeTracker.size
        self.measure = measure

    @staticmethod
    def size(hyperedge, subvalues):
        return sum(subvalues) + 1

    @staticmethod
    def depth(hyperedge, subvalues):
        return max([0] + subvalues) + 1

    def _update_node(self, node, to_update):
        for h in node.incoming:
            self._update_hyperedge(h, to_update)

    def _update_hyperedge(self, hyperedge, to_update):
        value = self.measure(hyperedge, [self.best.get(d, [self.worst_value])[0]
                                         for d in hyperedge.dst])
        src_best = self.best.setdefault(hyperedge.src, (self.worst_value, set()))
        if value < src_best[0]:
            self.best[hyperedge.src] = (value, {hyperedge})
            to_update.add(hyperedge.src)
        elif value == src_best[0]:
            src_best[1].add(hyperedge)
        elif hyperedge in src_best[1]:
            # tricky case: the size of the best hyperedge gets higher
            src_best[1].remove(hyperedge)
            if len(src_best[1]) == 0:
                self.best[hyperedge.src] = (self.worst_value, set())
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
            if isinstance(e, Node) and not e in self.best:
                self.best[e] = (self.worst_value, set())
        self._update(h for h in elements if isinstance(h, Hyperedge))



    def on_merge(self, hypergraph, node, removed, added):
        # If some of the merged (removed) hyperedges were among the best,
        # we should update them in the corresponding sets (except for the outgoings of `node')
        for h in removed:
            if h.src != node:
                h_src_best_set = self.best[h.src][1]
                if h in h_src_best_set:
                    h_src_best_set.remove(h)
                    h_src_best_set.add(h.merged)

        # Now take care of the outgoings of node
        n_best = self.best[node]
        m_best = self.best[node.merged]
        if n_best[0] < m_best[0]:
            self.best[node.merged] = (n_best[0], set(h.merged for h in n_best[1]))
            self._update([node.merged])
        elif n_best[0] == m_best[0]:
            m_best[1].update(set(h.merged for h in n_best[1]))

        del self.best[node]

        self._update(added)


    def on_remove(self, hypergraph, elements):
        for e in elements:
            assert isinstance(e, Hyperedge)
            if e.src in self.best:
                src_best = self.best[e.src]
                if e in src_best[1]:
                    if len(src_best[1]) > 1:
                        src_best[1].remove(e)
                    else:
                        self.best[e.src] = (self.worst_value, set())
                        self._update(list(e.src.outgoing) + [e.src])


    def on_remove_node(self, hypergraph, node, hyperedges):
        del self.best[node]
        self.on_remove(hypergraph, hyperedges)

    def best_terms(self, node):
        for h in self.best[node][1]:
            for subterms in itertools.product(*[self.best_terms(d) for d in h.dst]):
                yield Term(h.label, subterms)


def measure_term(term, function):
    return function(term.hyperedge, [measure_term(d, function) for d in term.dst])

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
