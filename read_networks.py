import networkx as nx
import polars as pl


def read_citation_graph(field, graph=nx.DiGraph) -> nx.DiGraph or nx.Graph:
    """
    Read the citation graph as (un)directed graph.
    Graph is weakly connected.

    :param filed: the field name such as 'History' and 'Physics'
    :param graph: the type of graph, directed or undirected. Default is directed graph.
    """
    citation_graph = nx.read_edgelist(
        f'data/{field}/citation_lcc.edgelist',
        create_using=graph
    )
    return citation_graph

