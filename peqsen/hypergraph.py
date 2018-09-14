
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


class Node:
    global_index = 0
    def __init__(self):
        self.incoming = set()
        self.outgoing = set()
        self.to_be_removed = False
        self.merged = None
        Node.global_index += 1
        self.index = Node.global_index

    def __repr__(self):
        return some_name(hash(self)) + str(self.index) + \
            ("(=" + repr(self.merged) + ")" if self.merged else "")

    def follow(self):
        return self if self.merged is None else self.merged.follow()


class Hyperedge:
    def __init__(self, label, src=None, dst=None):
        if dst is None:
            dst = []

        self.src = src
        self.dst = dst
        self.label = label
        self.to_be_removed = False
        self.merged = None

    def __repr__(self):
        return repr(self.src) + " -> " + repr(self.label) + " -> " + repr(self.dst) + \
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


class Hypergraph:
    def __init__(self):
        self._being_modified = False
        self._nodes = set()
        self._hyperedges = {}

    def __repr__(self):
        res = object.__repr__(self) + " {\n"
        for n in self._nodes:
            res += "    " + str(n) + " {\n"
            for h in n.outgoing:
                res += "        " + str(h) + "\n"
            res += "    }\n"
        res += "}\n"
        return res

    def nodes(self):
        return self._nodes

    def hyperedges(self):
        return self._hyperedges.values()

    def on_add(self, elements):
        pass

    def on_merge(self, node, added_hyperedges):
        pass

    def on_remove(self, elements):
        pass

    def rewrite(self, remove=(), add=(), merge=()):
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

        self._remove(remove, 0)

        added = []
        to_merge = list(merge)
        resmap = {}
        add_res = self._add(add, added, to_merge, resmap)
        self.on_add(added)

        removed = []
        self._remove(remove, 1, removed)
        self.on_remove(removed)

        self._merge(to_merge)

        self._being_modified = False

        return add_res

    def _remove(self, elements, phase, removed=None):
        for e in elements:
            if isinstance(e, Node):
                raise ValueError("Manual removal of nodes is forbidden")
            elif isinstance(e, Hyperedge):
                self._remove_hyperedge(e, phase, removed)
            else:
                raise ValueError("Cannot remove a thing of unknown type: {}".format(e))

    def _remove_node(self, node, phase):
        if node in self._nodes:
            for inc in node.incoming:
                self._remove_hyperedge(inc, phase)
            for out in node.outgoing:
                self._remove_hyperedge(out, phase)

            if phase == 0:
                node.to_be_removed = True
            elif phase == 1:
                if node.to_be_removed:
                    node.to_be_removed = False
                    self._nodes.remove(node)
        else:
            raise ValueError("Cannot remove a node which is not in the hypergraph: {}".format(node))

    def _remove_hyperedge(self, hyperedge, phase, removed):
        if phase == 0:
            if hyperedge.src is None or hyperedge.src not in self._nodes or \
               hyperedge not in hyperedge.src.outgoing or hyperedge.merged:
                raise ValueError("Hyperedge cannot be removed: {}", hyperedge)
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
            if n1 is not n2:
                # We cannot sort them like this, it breaks our implementation of cong closure
                # if len(n1.incoming) + len(n1.outgoing) > len(n2.incoming) + len(n2.outgoing):
                #     (n1, n2) = (n2, n1)
                n1.merged = n2
                added = []
                for h in list(itertools.chain(n1.incoming, n1.outgoing)):
                    if h.merged is None:
                        if self._hyperedges.get((h.label, tuple(h.dst))) is h:
                            del self._hyperedges[(h.label, tuple(h.dst))]
                        new_src = n2 if h.src is n1 else h.src
                        new_dst = [n2 if n is n1 else n for n in h.dst]
                        h.merged = self._add_hyperedge(Hyperedge(h.label, new_src, new_dst),
                                                       added, to_merge)
                        h.to_be_removed = True
                        self._remove_hyperedge(h, 1, [])
                self._nodes.remove(n1)
                self.on_merge(n1, added)
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
                        if h.src is not None and h.src is not e:
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
        existing = self._hyperedges.get((hyperedge.label, tuple(hyperedge.dst)))
        if existing is not None:
            if hyperedge.src == existing.src or hyperedge.src is None:
                existing.to_be_removed = False
                return existing
            else:
                to_merge.append((hyperedge.src, existing.src))
        else:
            self._hyperedges[(hyperedge.label, tuple(hyperedge.dst))] = hyperedge

        if hyperedge.src is None:
            hyperedge.src = Node()
            self._nodes.add(hyperedge.src)
            added.append(hyperedge.src)

        hyperedge.to_be_removed = False
        added.append(hyperedge)
        hyperedge.src.outgoing.add(hyperedge)
        for d in hyperedge.dst:
            d.incoming.add(hyperedge)

        return hyperedge

    def check_integrity(self, strict=True):
        """Sometimes hypergrah may be in a non-congruent-closed state. It may happen between
        on_add and on_merge when a node gets an outgoing hyperedge which leads to its merging, but
        it cannot be merged yet."""
        for n in self._nodes:
            assert n.merged is None
            for h in n.outgoing:
                assert h.merged is None
                assert h.src is n
                assert all(d in self._nodes for d in h.dst)
                assert all(h in d.incoming for d in h.dst)
                if strict:
                    assert self._hyperedges[(h.label, tuple(h.dst))] is h
            for h in n.incoming:
                assert h.merged is None
                assert h.src in self._nodes
                assert all(d in self._nodes for d in h.dst)
                assert all(h in d.incoming for d in h.dst)
                if strict:
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

@strategies.composite
def hyperedge_strategy(draw, nodes, labels=['a', 'b'], max_children=10):
    if isinstance(labels, (list, tuple)):
        labels = strategies.sampled_from(labels)
    if isinstance(nodes, (list, tuple)):
        nodes = strategies.sampled_from(nodes)
    src = draw(nodes)
    dst = draw(strategies.lists(nodes, max_size=max_children))
    label = draw(labels)
    return Hyperedge(label, src, dst)

@strategies.composite
def simple_addition_strategy(draw, labels=['a', 'b'],
                             max_nodes=20, max_hyperedges=100, max_children=10):
    nodes = [Node() for i in range(draw(strategies.integers(0, max_nodes)))]
    hyperedges = draw(strategies.lists(hyperedge_strategy(nodes, labels, max_children),
                                       max_size=max_hyperedges))
    return draw(strategies.permutations(nodes + hyperedges))

@strategies.composite
def hypergraph_strategy(draw, labels=strategies.sampled_from(['a', 'b']),
                        max_nodes=20, max_hyperedges=100, max_children=10):
    to_add = draw(simple_addition_strategy(labels=labels,
                                           max_nodes=max_nodes,
                                           max_hyperedges=max_hyperedges,
                                           max_children=max_children))
    h = Hypergraph()
    h.rewrite(add=to_add)
    return h

