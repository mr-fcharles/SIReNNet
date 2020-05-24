from SIReNet.sparse_graph import SparseGraph
import networkx as nx
import matplotlib.pyplot as plt


class SparseGraph_wutils(SparseGraph):

    '''
    Extends with utilities from networkX the base SparseGraph class

    Attributes
    ___________

    - graph: networkX undirected graph
        graph generated from the adjacency matrix

    - giant_component: networkX undirected graph
        giant component extracted from the generated network

    - connected: Boolean
        attributes that tells us whether the graph is connected or not

    Methods
    __________

    - build_nx_graph()
        Generates the graph from the sparse adjacency matrix using nx

    - is_connected()
        Tests whether the generated graph is connected

    - number_connected_components()
        Retruns the number of connected components

    - components_dimenison()
        Returns a list of the dimension of the connected compoments components of the graph in descending order

    -degree_distribution(retrun_value: boolean)
        Computes and plots the degree degree distribution of the graph. If return_value=True returns the raw data

    - build_giant_component(return_component: boolean)
        Stores the giant component in self.giant component. If return_component true returns the nx subgraph

    - plot_graph()
        Plots the whole graph

    - plot_giant_component()
        Plots the giant component

    '''

    def __init__(self,pop_size=1000,node_name='Paperopolis'):

        #inherits from SparseGraph
        super().__init__(pop_size,node_name)

        # NetworkX parameters
        self.graph = None
        self.giant_component = None
        self.connected = None

    ############################################################################


    def build_nx_graph(self):

        '''
        Generates the graph from the sparse adjacency matrix using nx
        :return: Stores nx graph in self.graph
        '''

        self.graph = nx.from_scipy_sparse_matrix(self.adjacency)

    ############################################################################

    def is_connected(self):
        '''
        Tests whether the generated graph is connected

        :return: stores the result of the test in self.connected
        '''
        try:

            if (self.connected is None):
                self.connected = nx.is_connected(self.graph)
                return self.connected

            else:
                return self.connected

        except:

            print('NetworkX graph not initialized. Run build_graph method')

    ###########################################################################

    def number_connected_components(self):

        '''
        Retruns the number of connected components
        :return: int
        '''

        # import networkx as nx

        try:

             return nx.number_connected_components(self.graph)

        except:

            print('NetworkX graph not initialized. Run build_graph method')

    #############################################################################

    def components_dimension(self):
        '''
        Returns a list of the dimension of the connected compoments components of the graph in descending order

        :return: list
        '''

        # import networkx as nx

        try:

            return [len(c) for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]

        except:

            print('NetworkX graph not initialized. Run build_graph method')

    #############################################################################

    def degree_distribution(self, return_values=False):
        '''
         Computes and plots the degree degree distribution of the graph. If return_value=True returns the raw data

        :param return_values: Bool
        :return: dictionary
        '''
        # import matplotlib.pyplot as plt
        import collections

        try:

            degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)  # degree sequence
            degreeCount = collections.Counter(degree_sequence)
            deg, cnt = zip(*degreeCount.items())

            fig, ax = plt.subplots()
            plt.bar(deg, cnt, width=0.80, color='b')

            plt.title("Degree Histogram")
            plt.ylabel("Count")
            plt.xlabel("Degree")
            ax.set_xticks([d + 0.4 for d in deg])
            ax.set_xticklabels(deg);

            if (return_values):
                out = {'degree': deg, 'count': cnt}

                return out

        except:

            print('NetworkX graph not initialized. Run build_graph method')



    #############################################################################

    def build_giant_component(self, return_component=False):
        '''
        Stores the giant component in self.giant component. If return_component true returns the nx subgraph

        :param return_component: Bool
        :return: nx.Graph instance
        '''
        try:

            self.giant_component = max(nx.connected_component_subgraphs(self.graph), key=len)

            if (return_component):
                return self.giant_component

        except:

            print('NetworkX graph not initialized. Run build_graph method')

    #############################################################################

    def plot_giant_component(self):
        '''
        Plots the whole graph
        '''
        try:

            nx.draw_spring(self.giant_component)

        except:

            print('Giant component initialized. Run giant_component method')

    #############################################################################

    def plot_graph(self):
        '''
        Plots the giant component
        :return:
        '''
        try:

            nx.draw_spring(self.graph)

        except:

            print('NetworkX graph not initialized. Run build_graph method')