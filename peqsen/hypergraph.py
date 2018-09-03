
import itertools

class Node:
    def __init__(self):
        self.incoming = []
        self.outgoing = []
        self.to_be_removed = False


class Hyperedge:
    def __init__(self, label, src=None, dst=None):
        if dst is None:
            dst = []

        self.src = src
        self.dst = dst
        self.label = label
        self.to_be_removed = False


class Term:
    def __init__(self, label, dst=None):
        if dst is None:
            dst = []

        self.dst = dst
        self.label = label
        self.hyperedge = None


class Hypergraph:
    def __init__(self):
        self._being_modified = False
        self._nodes = set()
        self._hyperedges = {}

    def nodes(self):
        return self._nodes

    def on_add(self, elements):
        pass

    def on_merge(self, node, added_hyperedges):
        pass

    def on_remove(self, elements):
        pass

    def rewrite(self, remove, add, merge):
        if self._being_modified:
            raise RuntimeError("Trying to modify a hypergraph during another modification")

        if isinstance(remove, [Hyperedge, Node]):
            remove = [remove]

        if isinstance(add, [Node, Hyperedge]):
            add = [add]

        self._being_modified = True

        self._remove(remove, phase=0)

        added = []
        to_merge = list(merge)
        resmap = {}
        add_res = self._add(add, added, to_merge, resmap)
        self.on_add(added)

        removed = []
        self._remove(remove, phase=1, removed)
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
                    self._nodes -= node
        else:
            raise ValueError("Cannot remove a node which is not in the hypergraph: {}".format(node))

    def _remove_hyperedge(self, hyperedge, phase, removed):
        if phase == 0:
            hyperedge.to_be_removed = True
        elif phase == 1:
            if hyperedge.to_be_removed:
                removed += hyperedge
                hyperedge.to_be_removed = False
                hyperedge.src.node.outgoing -= hyperedge
                for dmnode in hyperedge.dst:
                    dnode.node.incoming -= hyperedge
                self._hyperedges[(hyperedge.label, hyperedge.dst)] -= hyperedge

    def _merge(self, merge):
        if not merge:
            return
        to_merge = []
        for n1, n2 in merge:
            if n1 is not n2 and (n1.merged is None or n2.merged is None or
                                 n1.merged is not n2.merged):
                if len(n1.incoming) + len(n1.outgoing) > len(n2.incoming) + len(n2.outgoing):
                    (n1, n2) = (n2, n1)
                n1.merged = n2
                added = []
                for h in itertools.chain(n1.incoming, n1.outgoing):
                    if self._hyperedges.get((h.label, h.dst)) is h:
                        del self._hyperedges[(h.label, h.dst)]
                    new_src = n2 if h.src is n1 else h.src
                    new_dst = [n2 if n is n1 else n for n in h.dst]
                    h.merged = _add_hyperedge(Hyperedge(h.label, new_src, new_dst), added, to_merge)
                self._nodes.remove(n1)
                self.on_merge(n1, added)
        self._merge(to_merge)

    def _add(self, elements, added, to_merge, resmap):
        res = []
        for e in elements:
            if isinstance(e, Node):
                if e in self._nodes:
                    res.append(e)
                elif e in resmap:
                    n = resmap[e]
                    if n is None:
                        n = Node()
                        self._nodes += n
                        added.append(n)
                        resmap[e] = n
                    res.append(resmap[e])
                else:
                    resmap[e] = None
                    for h in e.outgoing:
                        if h.src is not None and h.src is not e:
                            raise ValueError("The source of a hyperedge being added is not the same"
                                             "as its parent node")
                        new_dst = self._add(h.dst, added, to_merge, resmap)
                        h_res = self._add_hyperedge(Hyperedge(h.label, resmap[e], new_dst),
                                                    added, to_merge, resmap)
                        resmap[e] = h_res.src
                        resmap[h] = h_res
                    res.append(resmap[e])
            elif isinstance(e, Hyperedge):
                if e in resmap:
                    res.append(resmap[e])
                else:
                    [new_src] = self._add([e.src], added, to_merge, resmap)
                    new_dst = self._add(e.dst, added, to_merge, resmap)
                    h = self._add_hyperedge(Hyperedge(e.label, new_src, new_dst),
                                            added, to_merge, resmap)
                    resmap[e] = h
                    res.append(e)
            else:
                raise ValueError("Cannot add this, unknown type: {}".format(e))
        return res

    def _add_hyperedge(self, hyperedge, added, to_merge):
        existing = self._hyperedges.get((hyperedge.label, hyperedge.dst))
        if existing is not None:
            if hyperedge.src == existing.src or hyperedge.src is None:
                existing.to_be_removed = False
                return existing
            else:
                to_merge.append((hyperedge.src, existing.src))
        else:
            self._hyperedges[(hyperedge.label, hyperedge.dst)] = hyperedge

        if hyperedge.src is None:
            hyperedge.src = Node()
            self._nodes += hyperedge.src
            added.append(hyperedge.src)

        hyperedge.to_be_removed = False
        added.append(hyperedge)
        hyperedge.src.outgoing += hyperedge
        for d in hyperedge.dst:
            d.incoming += hyperedge

        return hyperedge


