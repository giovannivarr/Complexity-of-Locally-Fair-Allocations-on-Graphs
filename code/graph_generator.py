import networkx as nx


def generate_matchings(n: int = 10) -> nx.Graph:
    """
    Generates a matching of size n.
    :param n: the (even) number of nodes of the output graph.
    :return: a graph G which is a matching of size n.
    """
    assert n % 2 == 0, 'n must be an even number'

    G = nx.Graph()
    G.add_nodes_from([x for x in range(n)])
    G.add_edges_from([(i, i + 1) for i in range(0, n, 2)])

    return G


def generate_line(n: int = 10) -> nx.Graph:
    """
    Generates a line of size n.
    :param n: the number of nodes of the output graph.
    :return: a graph G which is a line of size n.
    """

    G = nx.Graph()
    G.add_nodes_from([x for x in range(n)])
    G.add_edges_from([(i, i + 1) for i in range(n-1)])

    return G