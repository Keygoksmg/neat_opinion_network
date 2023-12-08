# %%
import networkx as nx
import numpy as np
import pandas as pd
import datetime
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# from utils import *
from read_networks import read_combined_graph_from_csv, read_citation_graph_from_edgelist


# %%
class GraphClass:
    """
    Class to set attributes matrix for experiment
    """
    # ==== Note: If you change a line here, please check if there's a corresponding line to be updated in the same way in ExperimentClass.add_edges_and_update ==== 
    def __init__(self, graph):
        # print(graph)
        self.graph = graph

        # create adjacency matrix
        self.adj_matrix = nx.to_numpy_array(self.graph)

        # == nodes ==
        # degree
        self.degrees = self.adj_matrix.sum(axis=1).reshape((-1, 1))

        # opinion
        self.opinions = np.array(list(nx.get_node_attributes(self.graph, "opinion").values())).reshape((-1, 1))

        # resistance
        self.resistance = np.array(list(nx.get_node_attributes(self.graph, "resistance").values())).reshape((-1, 1))

        # node_centrality
        self.node_centrality = np.array(list(nx.get_node_attributes(self.graph, "node_centrality").values())).reshape((-1, 1))
        
        # == edges ==
        # weight on edges
        self.weight = nx.attr_matrix(self.graph, edge_attr='weight')[0]

# %%
class GraphAttributeClass:
    """
    Class to add node/edge attributes to graph
    """
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)

    def assign_node_attr_opinion(self):
        """
        Initialize with a given timpstamp.
        If nodes are connected before this timestamp, they share a same init opinion.
        """
        for node in self.g.nodes:
            self.g.nodes[node]['opinion'] = 0

        # If nodes are connected before this timestamp, they share a same init opinion.
        for node in self.g.nodes:
            has_opinion_flag = False
            for init_nbr in self.g.neighbors(node):
                if self.g.nodes[init_nbr]['opinion'] != 0:
                    self.g.nodes[node]['opinion'] = self.g.nodes[init_nbr]['opinion']
                    has_opinion_flag = True
                    break
            if not has_opinion_flag:
                # use gaussian distribution
                self.g.nodes[node]['opinion'] = np.random.normal(0.5, 0.1)
            # print(self.g.nodes[node]['opinion'])

            # self.g.nodes[node]['resistance'] = np.random.uniform(0, 1)
            # self.g.nodes[node]['node_centrality'] = 0.01 + self.g.degree(node)

    def assign_node_attr_resistance(self, resistance_node_dict=None):
        if resistance_node_dict is None:
            for node in self.g.nodes:
                self.g.nodes[node]['resistance'] = np.random.uniform(0, 1)
        else:
            for node, value in resistance_node_dict.items():
                self.g.nodes[node]['resistance'] = value

    def assign_node_attr_centrality(self):
        for node in self.g.nodes:
            self.g.nodes[node]['node_centrality'] = 0.01 + self.g.degree(node)

    def assign_edge_attr_weight(self):
        for e1, e2 in self.g.edges:
            self.g.edges[e1, e2]['weight'] = np.random.rand()

    def add_attrs(self, g, resistance_node_dict=None, ):
        self.g = g
        self.assign_node_attr_opinion()
        self.assign_node_attr_resistance(resistance_node_dict)
        self.assign_node_attr_centrality()
        self.assign_edge_attr_weight()
        return self.g
    
    def add_recalculated_attrs(self, g):
        """
        Update node_centrality
        """
        self.g = g
        self.assign_node_attr_centrality()
        return self.g


# %%
class UpdateGraphClass(GraphAttributeClass):
    """
    Class to add edges and update the graph
    """
    def __init__(self, df: pd.DataFrame, current_date: str):
        super().__init__()
        self.df = df
        self.current_date = current_date

    def add_new_edges_per_time_step(self, g: nx.Graph | nx.DiGraph, mode: str='month') -> (list, str):
        def _get_next_month_first_day(datestr: str) -> str:
            daterep = datetime.datetime.strptime(datestr, '%Y-%m-%d').date()
            new_daterep = (daterep.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
            return str(new_daterep)

        def _extract_section_dataframe(df, start_date, end_date) -> pd.DataFrame:
            return df.loc[(df['CitationDate'] >= start_date) & (df['CitationDate'] < end_date)]
            
        # update mode: 'month' or 'year'
        if mode == 'month':
            new_date = _get_next_month_first_day(self.current_date)
        else:  # can add other time steps here (e.g. multiple months = run the function several times)
            new_date = self.current_date
        
        # extract new edges to add
        df_add = _extract_section_dataframe(self.df, self.current_date, new_date)
        
        # create new adding graph
        adding_graph = nx.from_pandas_edgelist(
            df_add,
            source = 'Citing_AuthorID',
            target = 'Cited_AuthorID',
            create_using = type(g)
        )
        # remove self-loop
        adding_graph.remove_edges_from(nx.selfloop_edges(adding_graph))
        adding_graph_with_attr = super().add_attrs(adding_graph)

        # merge with original graph
        graph_merged = nx.compose(adding_graph_with_attr, g)  # 

        # recalculate attrs
        graph_merged = super().add_recalculated_attrs(graph_merged)

        # update current date
        self.current_date = new_date

        # log
        # print('Added # edges', len(new_edgelist))
        # print(new_date)
        return graph_merged


# %%
class ExperimentClass(GraphClass, UpdateGraphClass):
    """
    Run simulation on a graph
    """
    def __init__(self, graph: nx.Graph, model_name: str, df: pd.DataFrame, init_date: str):
        # super().__init__(graph)
        GraphClass.__init__(self, graph)
        UpdateGraphClass.__init__(self, df=df, current_date=init_date)

        self.model_name = model_name
        self.opinions_per_iter = []
        self.avgdiff_per_iter = []
        self.opinions_std_per_iter = []
        
    def run_model(self, steps: int):
        self.initial_opinions = self.opinions.copy()

        for _ in range(1, steps):
            if self.model_name == 'Friedkin-Johnson':
                # since edge weights have not been well maintained, use node centrality nc_i as w_ii and nc_j as w_ij
                new_opinions = (self.adj_matrix @ (self.node_centrality * self.opinions) + self.node_centrality * self.initial_opinions) / (self.adj_matrix @ self.node_centrality + self.node_centrality + 0.001)
            elif self.model_name == 'Abebe':
                new_opinions = self.resistance * self.opinions + (1 - self.resistance) * (self.adj_matrix @ self.opinions) / (self.degrees + 0.001)
            elif self.model_name == 'New':
                new_opinions = self.resistance * self.opinions + (1 - self.resistance) * (self.adj_matrix @ (self.node_centrality * self.opinions)) / ((self.adj_matrix @ self.node_centrality) + 0.001)

            # calculate average difference from last iteration
            difference = np.abs(new_opinions - self.opinions)
            avg_difference = difference.sum() / len(self.opinions)
            
            # statistics of each iteration
            self.opinions_per_iter.append(new_opinions)
            self.avgdiff_per_iter.append(avg_difference)  # avg difference from last iter
            self.opinions_std_per_iter.append(np.std(new_opinions))  # std of opinions
            
            # update graph
            self._update_graph(new_opinions)

    def _update_graph(self, new_opinions):
        # update node attributes
        for i, node in enumerate(self.graph.nodes):
            # opinions
            self.graph.nodes[node]['opinion'] = new_opinions[i][0]

            # ==== TBD: also update susceptability at each iteration ====
        
        # add new edges by UpdateGraphClass
        self.graph = self.add_new_edges_per_time_step(self.graph)

        # update matrix (e.g. self.opinions, self.degrees, etc.)
        super().__init__(self.graph)  # previous code was: self.opinions = np.array(list(nx.get_node_attributes(self.graph, "opinion").values())).reshape((-1, 1))

    def plot(self):
        # plot each time of sum of opinions
        fig, ax = plt.subplots(figsize=(10, 5))
        #ax.plot(np.array(self.opinions_std_per_iter).sum(axis=1))
        ax.plot(np.array(self.opinions_std_per_iter))
        #print(np.array(self.opinions_per_iter).sum(axis=1))
        ax.set_xlabel('Time')
        ax.set_ylabel('Std of opinions')
        ax.set_title(f'Standard Deviation of opinions over time ({self.model_name})')
        plt.show()

    def get_opinions_per_iter(self):
        return self.opinions_per_iter
    def get_avgdiff_per_iter(self):
        return self.avgdiff_per_iter
    def get_opinions_std_per_iter(self):
        return self.opinions_std_per_iter
    def get_node_attr(self, node:int =None, attr: str=None) -> int:
        return self.graph.nodes[node][attr]