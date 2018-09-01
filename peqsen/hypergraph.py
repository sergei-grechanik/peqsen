
class Node:
    def __init__(self, data=None):
        if data is None:
            data = {}

        self.data = data
        self.incoming = []
        self.outgoing = []
        self.to_be_removed = False


class ModNode:
    def __init__(self, node, modifier):
        if isinstance(node, ModNode):
            self.node = node.node
            self.modifier = modifier.compose(node.modifier)
        else:
            self.node = node
            self.modifier = modifier

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.node == other.node and self.modifier == other.modifier
        else:
            return NotImplemented

    def __hash__(self):
        return (self.node, self.modifier).__hash__


class Hyperedge:
    def __init__(self, data=None, src=None, dst=None):
        if data is None:
            data = {}

        if dst is None:
            dst = []

        self.data = data
        self.src = src
        self.dst = dst
        self.to_be_removed = False


class HyperedgeAlgebra:
    def is_id(self, modifier):
        pass
    def compose(self, modifier1, modifier2):
        pass
    def pullback(self, modnode1, modnode2):
        pass
    def canonical(self, hyperedge):
        pass
    def hyperedge_by_rhs(self, data, dst):
        pass
    def label(self, hyperedge):
        pass
    def replace_nodes(self, hyperedge, nodemap):
        pass
    def merge_hyperedge_data(self, data1, data2):
        pass
    def merge_node_data(self, data1, data2):
        pass


class Hypergraph:
    def __init__(algebra):
        self.algebra = algebra
        self._being_modified = False
        self._nodes = set()
        self._hyperedges = {}

    def nodes(self):
        return self._nodes

    def rewrite(self, remove, add):
        if self._being_modified:
            raise RuntimeError("Trying to modify a hypergraph during another modification")

        if isinstance(remove, [Hyperedge, Node]):
            remove = [remove]

        if isinstance(add, [Node, Hyperedge]):
            add = [add]

        self._being_modified = True

        self._remove(remove, phase=0)
        self._add(add)
        self._remove(remove, phase=1)

        self._being_modified = False

    def _remove(self, elements, phase):
        for e in elements:
            if isinstance(e, Node):
                raise ValueError("Removal of nodes is forbidden")
            elif isinstance(e, Hyperedge):
                self._remove_hyperedge(e, phase)
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
                    self._nodes -= node
        else:
            raise ValueError("Cannot remove a node which is not in the hypergraph: {}".format(node))

    def _remove_hyperedge(self, hyperedge, phase):
        if phase == 0:
            hyperedge.to_be_removed = True
        elif phase == 1:
            if hyperedge.to_be_removed:
                hyperedge.src.node.outgoing -= hyperedge
                for dmnode in hyperedge.dst:
                    dnode.node.incoming -= hyperedge
                self._hyperedges[(self.algebra.label(hyperedge), hyperedge.dst)] -= hyperedge

    def _add(self, elements, resmap=None):
        if resmap is None:
            resmap = {}

        added = []
        for e in elements:
            if isinstance(e, Node):
                if e in self._nodes:
                    added.append(e)
                else:
                    for h in e.outgoing:
                        if h.src is not None and h.src.node is not None and h.src.node is not e:
                            raise ValueError("The source of a hyperedge being added is not the same"
                                             "as its parent node")
                        dst = [ModNode(d.modifier, self._add([d.node], resmap)) for d in h.dst]
                        h_already = self.get_hyperedge(data, dst)
                        if h_already is not None:
                            pass
                        else:
                            
            else:
                raise ValueError("Cannot add a non-term: {}".format(t))

    def _add_hyperedge(self, hyperedge, added):
        hc = self.algebra.canonical(h)
        hs = self._hyperedges.setdefault((self.algebra.label(hc), hc.dst), set())
        for existing in hs:
            if existing.src == hc.src:
                existing.to_be_removed = False
                self._assign_data(existing,
                                  self.algebra.merge_hyperedge_data(existing.data, hc.data))
                return
            else:
                mn1, mn2 = self.algebra.pullback(existing.src, hc.src)
                if self.algebra.is_id(mn1.modifier):
                    hc = self.algebra.replace_nodes(hc, {hc.src.node: mn2})
                    self._assign_data(existing,
                                      self.algebra.merge_hyperedge_data(existing.data, hc.data))
                    existing.to_be_removed = False
                    return
                else:
                    to_merge.append((existing.src, hc.src))

        if hc.src.node not in self._nodes:
            self._nodes += hyperedge.src.node
            added.append(h.src.node)

        hs += hc
        hc.to_be_removed = False
        added.append(hc)
        hc.src.node.outgoing += hc
        for mn in hc.src.dst:
            mn.node.incoming += hc
