
import peqsen.util

import attr
import itertools
import networkx
import hypothesis
import inspect

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
        if isinstance(other, type(self)):
            return self._global_index < other._global_index
        else:
            return NotImplemented

class Node(GloballyIndexed):
    def __init__(self):
        super().__init__()
        self.incoming = set()
        self.outgoing = set()
        self.to_be_removed = False
        self.merged = None

    def __repr__(self):
        return some_name(hash(self)) + str(self._global_index) + \
            ("(->" + repr(self.merged) + ")" if self.merged else "")

    def follow(self):
        return self if self.merged is None else self.merged.follow()

    def apply_map(self, mapping):
        res = mapping.get(self)
        return self.follow() if res is None else res.follow()

    def with_reason(self, reason):
        return self


class Hyperedge(GloballyIndexed):
    def __init__(self, label, src=None, dst=None, reason=None):
        super().__init__()

        if dst is None:
            dst = []

        self.src = src
        self.dst = dst
        self.label = label
        self.to_be_removed = False
        self.merged = None
        self.reason = reason

    def __repr__(self):
        label = self.label if isinstance(self.label, str) else repr(self.label)
        return "{" + some_name(hash(self), 3) + "} " + repr(self.src) + "=" + \
            label + repr(tuple(self.dst)) + \
            ("(->" + repr(self.merged) + ")" if self.merged else "")

    def incident(self, index):
        if index == -1:
            return self.src
        else:
            return self.dst[index]

    def follow(self, allow_copy=True):
        """Note that sometimes this function returns a completely new hyperedge. This is ok when we
        are adding it into a hypergraph, but there might be some corner cases, I'm not sure."""
        if self.merged is None:
            if allow_copy:
                if (self.src is None or self.src.merged is None) and \
                        all(d.merged is None for d in self.dst):
                    return self
                else:
                    return Hyperedge(self.label, None if self.src is None else self.src.follow(),
                                     [d.follow() for d in self.dst],
                                     reason=self.reason)
            else:
                return self
        else:
            return self.merged.follow()

    def apply_map(self, mapping):
        res = mapping.get(self)
        if res is None:
            return Hyperedge(self.label, None if self.src is None else self.src.apply_map(mapping),
                             [d.apply_map(mapping) for d in self.dst],
                             reason=self.reason)
        else:
            return res.follow()

    def with_reason(self, reason):
        return Hyperedge(self.label, self.src, self.dst, reason=reason)

@attr.s(slots=True, frozen=True, repr=False, cache_hash=True)
class Term:
    """A Term. `dst` must be a tuple of other terms or nodes"""
    label = attr.ib()
    dst = attr.ib(default=(), converter=tuple)

    def __repr__(self):
        args = "(" + str(self.dst[0]) + ")" if len(self.dst) == 1 else str(tuple(self.dst))
        if isinstance(self.label, str):
            return self.label + args
        else:
            return repr(self.label) + args

    def apply_map(self, mapping):
        if isinstance(self, Term):
            return Term(self.label, (d.apply_map(mapping) for d in self.dst))
        else:
            return self.apply_map(mapping)

def term(sexpr, dst=None):
    if dst is None:
        if isinstance(sexpr, Term) or isinstance(sexpr, Node):
            return sexpr
        elif isinstance(sexpr, (tuple, list)):
            return Term(sexpr[0], tuple(term(t) for t in sexpr[1:]))
        else:
            return Term(sexpr)
    else:
        return Term(sexpr, tuple(term(t) for t in dst))

def list_term_elements(term, top_node=None, reason=None, index=0):
    """Convert a term into a list of hyperedges and nodes."""
    if reason is None:
        reason = AddTermReason(term)

    if isinstance(term, Term):
        if top_node is None:
            n = Node()
            n._is_term_node = True
        else:
            n = top_node

        more_elements = []
        dst = []
        idx = index + 2
        for d in term.dst:
            subelements = list_term_elements(d, reason=reason, index=idx)
            more_elements.extend(subelements)
            dst.append(subelements[0])
            idx += len(subelements)

        h = Hyperedge(term.label, n, dst, reason=IthElementReason(reason, index + 1))
        if top_node is None:
            n.outgoing = (h,)
        for d in h.dst:
            if hasattr(d, '_is_term_node') and d._is_term_node:
                d.incoming.add(h)

        return [n, h] + more_elements
    elif isinstance(term, Node):
        if top_node is not None:
            raise ValueError("Cannot apply top_node to a trivial term")
        return [term]

def list_descendants(node, history=set()):
    if isinstance(node, Node):
        if node in history:
            return []
        new_history = history | {node}
        return [node] + [e for h in node.outgoing for e in list_descendants(h, new_history)]
    elif isinstance(node, Hyperedge):
        return [node] + [e for d in node.dst for e in list_descendants(d, history)]

@attr.s(slots=True, frozen=True)
class Equality:
    """Basically, a pair of Nodes or Terms."""
    name = attr.ib()
    lhs = attr.ib()
    rhs = attr.ib()
    vars2nodes = attr.ib()

def make_equality(func, name=None):
    names = [p for p in inspect.signature(func).parameters]
    vars2nodes = {n: Node() for n in names}
    pair = func(*(vars2nodes[n] for n in names))
    if len(pair) != 2:
        raise ValueError("make_equality: func must return a pair, not {}".format(pair))

    if name is None:
        name = str(pair[0]) + " = " + str(pair[1])

    return Equality(name=name, lhs=pair[0], rhs=pair[1], vars2nodes=vars2nodes)

@attr.s(slots=True, frozen=True)
class ByCongruence:
    """A reason for merging, contains two hyperedges."""
    lhs = attr.ib()
    rhs = attr.ib()

@attr.s(slots=True, frozen=True, cmp=False)
class ByRule:
    rule = attr.ib()
    match = attr.ib()

@attr.s(slots=True, frozen=True)
class AddTermReason:
    term = attr.ib()

@attr.s(slots=True, frozen=True)
class IthElementReason:
    reason = attr.ib()
    index = attr.ib()


class Listener:
    def on_add(self, hypergraph, elements):
        pass
    def on_merge(self, hypergraph, node, removed, added, reason):
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

    def on_merge(self, node, removed, added, reason):
        for l in self.listeners:
            l.on_merge(self, node, removed, added, reason)

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

        # Transform Terms into nodes and hyperedges
        add_copy = add
        add = []
        post_add = []
        for e in add_copy:
            if isinstance(e, Term):
                term_elements = list_term_elements(e)
                add.append(term_elements[0])
                post_add.extend(term_elements[1:])
            else:
                add.append(e)
        add.extend(post_add)

        remove = [e.follow() for e in remove]
        add = [e.follow() for e in add]
        # Convert pairs to triples, the third element is an optional reason
        merge = [(es[0].follow(), es[1].follow(), (None if len(es) <= 2 else es[2]))
                 for es in merge]

        self._being_modified = True

        self._remove(remove, 0, ignore_already_removed=ignore_already_removed)

        added = []
        to_merge = merge
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
        for n1, n2, reason in merge:
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
                            new_h = Hyperedge(h.label, new_src, new_dst,
                                              reason=h.reason)
                            new_h.to_be_removed = h.to_be_removed
                            h.merged = self._add_hyperedge(new_h, added, to_merge)
                        h.to_be_removed = True
                        self._remove_hyperedge(h, 1, removed)
                self._nodes.remove(n1)
                self.on_merge(n1, removed, added, reason)
        self._merge(to_merge)

    def _add(self, elements, added, to_merge, resmap):
        """Add `elements` which may be nodes or hyperedges
        The result is a list of corresponding added (or already existing) elements
        - `added` is augmented with newly added stuff (which didn't exist in the graph before)
        - `to_merge` is augmented with pairs of nodes to merge as a result of this addition
        - `resmap` is augmented with mapping from the given elements to the corresponding graph
          elements."""
        res = []
        for e in elements:
            if isinstance(e, Node):
                if e in self._nodes:
                    # If it is already in the graph, nothing to do
                    res.append(e)
                elif e in resmap:
                    n = resmap[e]
                    res.append(resmap[e])
                else:
                    n = Node()
                    self._nodes.add(n)
                    resmap[e] = n
                    added.append(n)
                    res.append(n)
            elif isinstance(e, Hyperedge):
                new_dst = self._add(e.dst, added, to_merge, resmap)

                if e.src is None or e.src in self._nodes:
                    h = self._add_hyperedge(Hyperedge(e.label, e.src, new_dst,
                                                      reason=e.reason),
                                            added, to_merge)
                elif e.src in resmap:
                    h = self._add_hyperedge(Hyperedge(e.label, resmap[e.src], new_dst,
                                                      reason=e.reason),
                                            added, to_merge)
                else:
                    h = self._add_hyperedge(Hyperedge(e.label, None, new_dst,
                                                      reason=e.reason),
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
                    to_merge.append((hyperedge.src, existing.src,
                                     ByCongruence(hyperedge, existing)))

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

def gen_number(descr):
    if isinstance(descr, int):
        return strategies.just(descr)
    elif isinstance(descr, tuple) and len(descr) == 2:
        return strategies.integers(descr[0], descr[1])
    elif isinstance(descr, (list, set)):
        return strategies.sampled_from(descr)
    else:
        return descr

@strategies.composite
def gen_list(draw, value_strategy, size_descr):
    if not value_strategy:
        if isinstance(size_descr, int):
            if size_descr == 0:
                return []
        elif isinstance(size_descr, tuple) and len(size_descr) == 2:
            if size_descr[0] <= 0 <= size_descr[1]:
                return []
        elif isinstance(size_descr, (list, set)):
            if 0 in size_descr:
                return []
        else:
            size_descr = gen_number(size_descr)
            for _ in range(10):
                if draw(size_descr) == 0:
                    return []

        hypothesis.assume(False)

    if isinstance(value_strategy, (list, set)):
        value_strategy = strategies.sampled_from(value_strategy)

    # Note that generating a list directly is more efficient than generating a size and then
    # a list separately (probably because they use some smart heuristic for average list size)
    if isinstance(size_descr, tuple) and len(size_descr) == 2:
        min_size, max_size = size_descr
    else:
        min_size = max_size = draw(gen_number(size_descr))

    return draw(strategies.lists(value_strategy, min_size=min_size, max_size=max_size))

DEFAULT_SIGNATURE={'A': (0, 4), 'B': (0, 4), 'C': (0, 0)}

def leaf_labels(signature):
    res = []
    for k, v in signature.items():
        if v == 0 or (isinstance(v, tuple) and v[0] <= 0):
            res.append(k)
    return sorted(res)

def nonleaf_subsignature(signature):
    res = {}
    for k, v in signature.items():
        if isinstance(v, tuple) and v[1] > 0:
            res[k] = (max(v[0], 1), v[1])
        elif isinstance(v, int) and v > 0:
            res[k] = v
    return res

@strategies.composite
#@peqsen.util.traced
def gen_hyperedge(draw, nodes, signature=DEFAULT_SIGNATURE, acyclic=False):
    if isinstance(nodes, (list, tuple)):
        nodes_strat = strategies.sampled_from(nodes)
    else:
        nodes_strat = nodes

    if isinstance(signature, dict):
        label = draw(strategies.sampled_from(sorted(signature.keys())))
        dst_size_strat = gen_number(signature[label])

    src = draw(nodes_strat)

    if acyclic:
        # Sometimes src is the minimal node which might be very bad, decrease chance of this
        src_alt = draw(nodes_strat)
        src = max(src, src_alt)
        if isinstance(nodes, (list, tuple)):
            filtnodes = [n for n in nodes if n < src]
            dst = draw(gen_list(filtnodes, dst_size_strat))
        else:
            dst = draw(gen_list(nodes_strat.filter(lambda n: n < src), dst_size_strat))
    else:
        dst = draw(gen_list(nodes_strat, dst_size_strat))

    return Hyperedge(label, src, dst)

@strategies.composite
#@peqsen.util.traced
def gen_simple_addition(draw, signature=DEFAULT_SIGNATURE,
                        num_nodes=(0, 10), num_hyperedges=(0, 100),
                        acyclic=False):
    nodes = [Node() for i in range(draw(gen_number(num_nodes)))]
    if not nodes:
        return nodes
    hyperedges = draw(gen_list(gen_hyperedge(nodes, signature, acyclic=acyclic), num_hyperedges))
    return nodes + hyperedges

@strategies.composite
#@peqsen.util.traced
def gen_rewrite(draw, hypergraph, signature=DEFAULT_SIGNATURE,
                num_add_nodes=(0, 5), num_add_hyperedges=(0, 20),
                num_remove=(0, 10), num_merge=(0, 10)):
    add_nodes = [Node() for i in range(draw(gen_number(num_add_nodes)))]
    nodes = list(hypergraph.nodes())
    hyperedges = list(hypergraph.hyperedges())

    add_hyperedges = []
    remove = []
    merge = []

    if nodes or add_nodes:
        add_hyperedges = draw(gen_list(gen_hyperedge(add_nodes + nodes,
                                                     signature=signature),
                                       num_add_hyperedges))

    if nodes or hyperedges:
        remove = draw(gen_list(hyperedges, num_remove))

    if nodes:
        merge = draw(gen_list(strategies.tuples(strategies.sampled_from(nodes),
                                                strategies.sampled_from(nodes)),
                              num_merge))

    if draw(strategies.booleans()):
        add_size = len(add_nodes) + len(add_hyperedges)
        add = draw(gen_list(add_nodes + add_hyperedges, add_size))
    else:
        add = draw(strategies.permutations(add_nodes + add_hyperedges))

    return {'remove': remove, 'add': add, 'merge': merge}

@strategies.composite
#@peqsen.util.traced
def gen_permuted_rewrite(draw, rewrite):
    res = {}
    for key in rewrite:
        res[key] = draw(strategies.permutations(rewrite[key]))
    return res

@strategies.composite
#@peqsen.util.traced
def gen_hypergraph(draw, signature=DEFAULT_SIGNATURE,
                   num_nodes=(0, 10), num_hyperedges=(0, 100), congruence=True):
    to_add = draw(gen_simple_addition(signature=signature,
                                      num_nodes=num_nodes,
                                      num_hyperedges=num_hyperedges))
    h = Hypergraph(congruence=congruence)
    h.rewrite(add=to_add)
    return h


# TODO: Currently the second component, the hypergraph, may contain non-descendants of the root
@strategies.composite
#@peqsen.util.traced
def gen_pattern(draw, signature=DEFAULT_SIGNATURE, num_nodes=(1, 10), num_hyperedges=(1, 10)):
    """A pattern is a rooted acyclic (no directed cycles) hypergraph, possibly
    non-congruently-closed. Note that each sample is a pair (Node, Hypergraph).
    Moreover, the node is guaranteed to have at least one successor."""
    to_add = draw(gen_simple_addition(signature=signature,
                                      num_nodes=num_nodes,
                                      num_hyperedges=num_hyperedges,
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

@strategies.composite
#@peqsen.util.traced
def gen_term(draw, signature=DEFAULT_SIGNATURE, max_leaves=20, max_variables=5,
             variables=None, equality=False):
    if variables is None:
        vars_number = draw(strategies.integers(0, max_variables))
        variables = [Node() for _ in range(vars_number)]
    leaves = [term(l, ()) for l in leaf_labels(signature)] + variables

    @strategies.composite
    #@peqsen.util.traced
    def _extend_term(draw, subgen, sig=nonleaf_subsignature(signature)):
        label = draw(strategies.sampled_from(sorted(sig.keys())))
        child_len = draw(gen_number(sig[label]))
        return term(label, draw(strategies.lists(subgen, min_size=child_len, max_size=child_len)))

    gen = strategies.recursive(strategies.sampled_from(leaves), _extend_term)
    return draw(strategies.tuples(gen, gen)) if equality else draw(gen)
