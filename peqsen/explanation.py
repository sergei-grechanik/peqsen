
import itertools
import collections
import attr

from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, TriggerManager, parse, still_match

IncidentNode = attr.make_class('IncidentNode', ['hyperedge', 'rhs'], frozen=True)
IncidentNode.__doc__ = """A reason for merging, contains two hyperedges."""


class Point:
    def __init__(self):
        self.merged = None
        self.merged_reason = None

class ExplanationTracker(Listener):
    def __init__(self, hypergraph):
        self.node_points = {}
        self.hyperedge_points = {}

    def on_add(self, hypergraph, elements):
        for e in elements:
            if isinstance(e, Node):
                self.node_points[e] = Point()
        for e in elements:
            if isinstance(e, Hyperedge):
                src_point = Point(IncidentNode(e.reason), merged=self.node_points[e.src])
                dst_points = [Point(IncidentNode(e.reason, i), merged=self.node_points[d])
                              for i, d in enumerate(e.dst)]
                self.hyperedge_points[e] = [(e.reason, src_point, dst_points)]

    def on_merge(self, hypergraph, node, removed, added, reason):
        p1 = self.node_points[node]
        p2 = self.node_points[node.merged]
        p = Point()

        p1.merged = p
        p2.merged = p
        p1.merge_reason = reason
        p2.merge_reason = reason

        self.node_points[node.merged] = p
        del self.node_points[node]

        for h in removed:
            self.hyperedge_points.setdefault(h.merged, []).extend(self.hyperedge_points.get(h, []))

    def explain(self, point1, point2):
        pass
