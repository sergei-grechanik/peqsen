
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

    def __lt__(self, other):
        return self._global_index < other._global_index

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
        self._hash = hash((self.label, tuple(self.dst)))

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
        return self is other or (self.label == other.label and self.dst == other.dst)

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        return self.label < other.label or tuple(self.dst) < tuple(other.dst)


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
    def __init__(self, congruence=True):
        self._being_modified = False
        self._nodes = set()
        self._hyperedges = {}
        self._congruence = congruence
        self.listeners = set()

    def __repr__(self):
        cong = "" if self._congruence else " NONCONG"
        res = object.__repr__(self) + cong + " {\n"
        for n in self._nodes:
            res += "    " + str(n) + " {\n"
            for h in n.outgoing:
                if self._congruence:
                    mark = '' if self._hyperedges.get((h.label, tuple(h.dst))) is h else '! '
                else:
                    mark = '' if h in self._hyperedges.get((h.label, tuple(h.dst))) else '! '
                res += "        " + mark + str(h) + "\n"
            res += "    }\n"
        res += "}"
        return res

    def nodes(self):
        return self._nodes

    def hyperedges(self):
        if self._congruence:
            return list(self._hyperedges.values())
        else:
            # We have to create a list because itertools creates an iterator
            return list(itertools.chain(*self._hyperedges.values()))

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

        if isinstance(add, (Node, Hyperedge, Term, Recursively)):
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
                if self._congruence:
                    if self._hyperedges.get((hyperedge.label, tuple(hyperedge.dst))) is hyperedge:
                        del self._hyperedges[(hyperedge.label, tuple(hyperedge.dst))]
                else:
                    lst = self._hyperedges.get((hyperedge.label, tuple(hyperedge.dst)))
                    if lst and hyperedge in lst:
                        lst.remove(hyperedge)

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

                        if self._congruence:
                            if self._hyperedges.get((h.label, tuple(h.dst))) is h:
                                # bad case: merging of a hyperedge which is the main one
                                del self._hyperedges[(h.label, tuple(h.dst))]
                        else:
                            if h in self._hyperedges.get((h.label, tuple(h.dst))):
                                self._hyperedges.get((h.label, tuple(h.dst))).remove(h)

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
        """Add `elements` which may be nodes, hyperedges, terms and elements to add recursively
        The result is a list of corresponding added (or already existing) elements
        - `added` is augmented with newly added stuff (which didn't exist in the graph before)
        - `to_merge` is augmented with pairs of nodes to merge as a result of this addition
        - `resmap` is augmented with mapping from the given elements (or their descendants in
          the case of Recursive) to the corresponding graph elements."""
        res = []
        for e in elements:
            recursively = False
            if isinstance(e, Recursively):
                recursively = True
                e = e.node

            if isinstance(e, Node):
                if e in self._nodes:
                    # If it is already in the graph, nothing to do, even if `recursively`,
                    # because in this case all its descendants are in the graph too
                    res.append(e)
                elif e in resmap:
                    n = resmap[e]
                    if n is None:
                        # This case means that we are adding the node recursively and we reached
                        # the same node along some cyclic path, but we haven't
                        # added any corresponding node yet, so just add a node
                        n = Node()
                        self._nodes.add(n)
                        added.append(n)
                        resmap[e] = n
                    res.append(resmap[e])
                elif recursively and len(e.outgoing) > 0:
                    # Note that if there are no outgoing hyperedges then there's nothing to do
                    # recursively
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
        if self._congruence:
            existing = self._hyperedges.get((hyperedge.label, tuple(hyperedge.dst)))
            if existing is not None:
                if hyperedge.src == existing.src or hyperedge.src is None:
                    # Good, nothing to do, just mark that we don't want to remove it
                    existing.to_be_removed = existing.to_be_removed and hyperedge.to_be_removed
                    return existing
                else:
                    # in this case we have to merge the sources, and continue adding the hyperedge
                    to_merge.append((hyperedge.src, existing.src))

            # Sometimes there may be an existing hyperedge which wasn't registered in _hyperedges
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

        if self._congruence:
            # Register the hyperedge if it wasn't registered before
            if existing is None:
                self._hyperedges[(hyperedge.label, tuple(hyperedge.dst))] = hyperedge
        else:
            lst = self._hyperedges.setdefault((hyperedge.label, tuple(hyperedge.dst)), [])
            lst.append(hyperedge)

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
                    if self._congruence:
                        assert self._hyperedges[(h.label, tuple(h.dst))] is h
                    else:
                        assert h in self._hyperedges[(h.label, tuple(h.dst))]
            for h in n.incoming:
                assert h.merged is None
                assert h.src in self._nodes
                assert all(d in self._nodes for d in h.dst)
                assert all(h in d.incoming for d in h.dst)
                if strict:
                    assert not h.to_be_removed
                    if self._congruence:
                        assert self._hyperedges[(h.label, tuple(h.dst))] is h
                    else:
                        assert h in self._hyperedges[(h.label, tuple(h.dst))]
        for h in self.hyperedges():
            assert h.src in self._nodes

    def as_networkx(self):
        graph = networkx.MultiDiGraph()
        graph.add_nodes_from(id(n) for n in self.nodes())
        graph.add_nodes_from((id(h), {'label': h.label}) for h in self.hyperedges())
        for h in self.hyperedges():
            graph.add_edge(id(h.src), id(h))
            for i, d in enumerate(h.dst):
                graph.add_edge(id(h), id(d), key=i, idx=i)
        return graph

    def isomorphic(self, other):
        return networkx.algorithms.isomorphism.is_isomorphic(self.as_networkx(),
                                                             other.as_networkx(),
                                                             node_match=lambda x, y: x == y,
                                                             edge_match=lambda x, y: x == y,)

def map_rewrite(rewrite, mapping):
    return {'remove': [mapping[h] for h in rewrite['remove']],
            'add': [x.apply_map(mapping) for x in rewrite['add']],
            'merge': [tuple(mapping[x] for x in p) for p in rewrite['merge']]}

DEFAULT_SIGNATURE={label: strategies.integers(0, 4) for label in ['A', 'B', 'C']}

@strategies.composite
def gen_hyperedge(draw, nodes, signature=DEFAULT_SIGNATURE, acyclic=False):
    if isinstance(signature, dict):
        label = draw(strategies.sampled_from(sorted(signature.keys())))
        dst_size = draw(signature[label])
    if isinstance(nodes, (list, tuple)):
        nodes_strat = strategies.sampled_from(nodes)
    else:
        nodes_strat = nodes
    src = draw(nodes_strat)
    if acyclic:
        if isinstance(nodes, (list, tuple)):
            filtnodes = [n for n in nodes if n < src]
            dst = draw(strategies.lists(strategies.sampled_from(filtnodes),
                                        max_size=dst_size, min_size=dst_size))
        else:
            dst = draw(strategies.lists(nodes_strat.filter(lambda n: n < src),
                                        max_size=dst_size, min_size=dst_size))
    else:
        dst = draw(strategies.lists(nodes_strat, max_size=dst_size, min_size=dst_size))
    return Hyperedge(label, src, dst)

@strategies.composite
def gen_simple_addition(draw, signature=DEFAULT_SIGNATURE,
                        min_nodes=0, max_nodes=10, min_hyperedges=0,
                        max_hyperedges=100, acyclic=False):
    nodes = [Node() for i in range(draw(strategies.integers(min_nodes, max_nodes)))]
    hyperedges = draw(strategies.lists(gen_hyperedge(nodes, signature, acyclic=acyclic),
                                       min_size=min_hyperedges,
                                       max_size=max_hyperedges))
    return nodes + hyperedges

@strategies.composite
def gen_rewrite(draw, hypergraph, signature=DEFAULT_SIGNATURE,
                max_add_nodes=10, min_add_hyperedges=0, max_add_hyperedges=20,
                max_remove=20, max_merge=20):
    add_nodes = [Node() for i in range(draw(strategies.integers(0, max_add_nodes)))]
    nodes = list(hypergraph.nodes())
    add_hyperedges = draw(strategies.lists(gen_hyperedge(add_nodes + nodes, signature=signature),
                                           min_size=min_add_hyperedges,
                                           max_size=max_add_hyperedges))
    hyperedges = list(hypergraph.hyperedges())
    remove = draw(strategies.lists(strategies.sampled_from(hyperedges), max_size=max_remove))
    merge = draw(strategies.lists(strategies.tuples(strategies.sampled_from(nodes),
                                                    strategies.sampled_from(nodes)),
                                  max_size=max_merge))

    if draw(strategies.booleans()):
        add_size = len(add_nodes) + len(add_hyperedges)
        add = draw(strategies.lists(strategies.sampled_from(add_nodes + add_hyperedges),
                                    max_size=add_size, min_size=add_size))
    else:
        add = draw(strategies.permutations(add_nodes + add_hyperedges))

    return {'remove': remove, 'add': add, 'merge': merge}

@strategies.composite
def gen_permuted_rewrite(draw, rewrite):
    res = {}
    for key in rewrite:
        res[key] = draw(strategies.permutations(rewrite[key]))
    return res

@strategies.composite
def gen_hypergraph(draw, signature=DEFAULT_SIGNATURE,
                   min_nodes=0, max_nodes=10, max_hyperedges=100, congruence=True):
    to_add = draw(gen_simple_addition(signature=signature,
                                      min_nodes=min_nodes,
                                      max_nodes=max_nodes,
                                      max_hyperedges=max_hyperedges))
    h = Hypergraph(congruence=congruence)
    h.rewrite(add=to_add)
    return h


# TODO: Currently the second component, the hypergraph, may contain non-descendants of the root
@strategies.composite
def gen_pattern(draw, signature=DEFAULT_SIGNATURE, max_nodes=5, max_hyperedges=20):
    """A pattern is a rooted acyclic (no directed cycles) hypergraph, possibly
    non-congruently-closed. Note that each sample is a pair (Node, Hypergraph).
    Moreover, the node is guaranteed to have at least one successor."""
    to_add = draw(gen_simple_addition(signature=signature,
                                      min_nodes=1,
                                      max_nodes=max_nodes,
                                      min_hyperedges=1,
                                      max_hyperedges=max_hyperedges,
                                      acyclic=True))
    maxnode = max(n for n in to_add if isinstance(n, Node))
    maxnode_index = to_add.index(maxnode)
    h = Hypergraph(congruence=False)
    added = h.rewrite(add=to_add)
    added_maxnode = added[maxnode_index]
    if added_maxnode.outgoing:
        return added[maxnode_index], h
    else:
        nonempty_nodes = [n for n in h.nodes() if n.outgoing]
        hypothesis.assume(nonempty_nodes)
        maxnode = max(nonempty_nodes)
        return maxnode, h
