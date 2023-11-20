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


def read_combined_graph_from_csv(
    field, \
    filename = 'ssn_author_ref_combined.csv', \
    init_cutoff_date = '2019-03-31', \
    simplified = False, \
    graph=nx.DiGraph) -> nx.DiGraph or nx.Graph:
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

    simplified_citation_graph = nx.from_pandas_edgelist(
        df,
        source = 'Citing_AuthorID',
        target = 'Cited_AuthorID',
    )
    simplified_citation_graph.remove_edges_from(nx.selfloop_edges(simplified_citation_graph))

    cutoff_df = df.loc[df['CitationDate'] <= init_cutoff_date]
    #print(cutoff_df)
    initial_citation_graph = nx.from_pandas_edgelist(
        cutoff_df,
        source = 'Citing_AuthorID',
        target = 'Cited_AuthorID',
    )
    initial_citation_graph.add_nodes_from(simplified_citation_graph)
    initial_citation_graph.remove_edges_from(nx.selfloop_edges(initial_citation_graph))

    return simplified_citation_graph, initial_citation_graph, df
