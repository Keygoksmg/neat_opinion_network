import networkx as nx
import pandas as pd


def read_citation_graph_from_edgelist(field, filename='citation_lcc.edgelist', graph=nx.DiGraph) -> nx.DiGraph or nx.Graph:
    """
    Read the citation graph as (un)directed graph, based on an nx edgelist file.
    Graph is weakly connected.

    :param filed: the field name such as 'History' and 'Physics'
    :param filename: the name of the edge list. Default is 'citation_lcc.edgelist'.
    :param graph: the type of graph, directed or undirected. Default is directed graph.
    """
    citation_graph = nx.read_edgelist(
        f'data/{field}/{filename}',
        create_using=graph
    )
    return citation_graph


# extract connected component
def extract_largest_connected_component(g):
    if g.is_directed():
        lcc = max(nx.weakly_connected_components(g), key=len)
        g_lcc = g.subgraph(lcc).copy()
    else:
        lcc = max(nx.connected_components(g), key=len)
        g_lcc = g.subgraph(lcc).copy()
    return g_lcc


def read_combined_graph_from_csv(
    field,
    filename = 'ssn_author_ref_combined.csv',
    init_cutoff_date = '2019-03-31',
    simplified = False,
    graph=nx.Graph()) -> nx.DiGraph | nx.Graph:
    """
    (from Shengqi-11/19 update)
    Read the citation graph as (un)directed graph, based on an csv file.
    Graph is weakly connected. File is already sorted by date.
    Returns:
    (1) a simplified graph with all final edges,
    (2) (if simplified = False; else None) initialized graph based on connections before the cutoff data, 
    (3) the dataframe itself.

    :param filed: the field name such as 'History' and 'Physics'
    :param filename: the name of the edge list. Default is 'ssn_author_ref_combined.csv'.
    :param init_cutoff_date: use the citations before this date (i.e., the first few rows) to initialize the graph.
    :param simplied: a flag of whether we omit the time-step edge update. Default is False.
    :param graph: the type of graph, directed or undirected. Default is directed graph.
    """
    df = pd.read_csv(f'data/{field}/{filename}', sep=',')

    full_graph = nx.from_pandas_edgelist(
        df,
        source = 'Citing_AuthorID',
        target = 'Cited_AuthorID',
        create_using = graph
    )
    # remove self-loops
    full_graph.remove_edges_from(nx.selfloop_edges(full_graph))
    # extract LCC as graph
    full_grpah_lcc = extract_largest_connected_component(full_graph)

    # extract all nodes in LCC as df at final step
    nodes_full_lcc = list(full_grpah_lcc.nodes())
    df_with_lcc = df[
        df['Citing_AuthorID'].isin(nodes_full_lcc) &
        df['Cited_AuthorID'].isin(nodes_full_lcc)
    ]

    # get df with cutoff date
    cutoff_df = df_with_lcc.loc[df_with_lcc['CitationDate'] <= init_cutoff_date]
    initial_graph = nx.from_pandas_edgelist(
        cutoff_df,
        source = 'Citing_AuthorID',
        target = 'Cited_AuthorID',
        create_using = graph
    )
    # remove self-loop
    initial_graph.add_nodes_from(full_grpah_lcc)
    initial_graph.remove_edges_from(nx.selfloop_edges(initial_graph))

    return full_grpah_lcc, initial_graph, init_cutoff_date, df_with_lcc
