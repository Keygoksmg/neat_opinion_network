import datetime
import pandas as pd
import networkx as nx

def section_dataframe(df, start_date, end_date):
    return df.loc[(df['CitationDate'] >= start_date) & (df['CitationDate'] < end_date)]

def df_to_list_of_edges(df):
    temp_graph = nx.from_pandas_edgelist(df, source = 'Citing_AuthorID', target = 'Cited_AuthorID')
    return temp_graph.edges()

def get_next_month_first_day(datestr: str) -> str:
    daterep = datetime.datetime.strptime(datestr, '%Y-%m-%d').date()
    new_daterep = (daterep.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    return str(new_daterep)