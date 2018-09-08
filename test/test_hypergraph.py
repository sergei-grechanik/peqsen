from peqsen.hypergraph import Hypergraph, Node, Hyperedge

def test_basic_operations():
    hg = Hypergraph()
    print(hg.as_text())
    [ha] = hg.rewrite(add=Hyperedge('a', dst=[]))
    print(hg.as_text())
    [hb1, hb2] = hg.rewrite(add=[Hyperedge('b', dst=[ha.src]),
                                 Hyperedge('b', dst=[ha.src, ha.src])])
    print(hg.as_text())
    [hc1, hc2] = hg.rewrite(add=[Hyperedge('c', dst=[hb1.src]),
                                 Hyperedge('c', dst=[hb2.src])])
    print(hg.as_text())
    hg.rewrite(merge=[(hb1.src, hb2.src)])
    print(hg.as_text())
    hg.rewrite(remove=[hb1])
    print(hg.as_text())

if __name__ == "__main__":
    test_basic_operations()
