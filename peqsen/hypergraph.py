
class Node:
    def __init__(self, data=None):
        if data is None:
            data = {}

        self.data = data
        self.incoming = []
        self.outgoing = []


class Hyperedge:
    def __init__(self, data=None):
        if data is None:
            data = {}

        self.data = data
        self.src = None
        self.dst = []


class Term:
    def __init__(self, data=None, dst=None):
        if data is None:
            data = {}

        if dst is None:
            dst = []

        self.data = data
        self.dst = dst


class Hypergraph:
    def __init__():
        self._being_modified = False
        self.nodes = set()

    def nodes(self):
        return self.nodes

    def remove(self, elements):
        if self._being_modified:
            raise RuntimeError("Trying to modify a hypergraph during another modification")

        if isinstance(elements, [Hyperedge, Node]):
            return self.remove([elements])

        self._being_modified = True

        for e in elements:
            if isinstance(e, Node):
                self._remove_node(e)
            elif isinstance(e, Hyperedge):
                self._remove_hyperedge(e)
            else:
                raise ValueError("Cannot remove a thing of unknown type: {}".format(e))

        self._being_modified = False

    def _remove_node(self, node):
        if node in self.nodes:
            for inc in node.incoming:
                self._remove_hyperedge(inc)
            for out in node.outgoing:
                self._remove_hyperedge(out)
            self.nodes -= node
        else:
            raise ValueError("Cannot remove a node which is not in the hypergraph: {}".format(node))

    def _remove_hyperedge(self, hyperedge):
        hyperedge.src.outgoing -= hyperedge
        for dnode in hyperedge.dst:
            dnode.incoming -= hyperedge

    def add(self, terms):
        if self._being_modified:
            raise RuntimeError("Trying to modify a hypergraph during another modification")

        if isinstance(terms, Term):
            return self.add([terms])

        self._being_modified = True

        for t in terms:
            if isinstance(t, Term):
                self._add_term(t)
            else:
                raise ValueError("Cannot add a non-term: {}".format(t))

        self._being_modified = False

    def _add_term(self, term):
        dst = [ for sub in term.dst]


def stuff():
    pass
