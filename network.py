import igraph
from igraph import *
from nltk import bigrams
import numpy as np
from sklearn.metrics import pairwise_distances
import codecs
import platform
import os
import subprocess
import pandas as pd
from scipy import integrate

def get_largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def generate_xnet(graph, path):
    result = "#vertices " + str(graph.vcount()) + " nonweighted \n"
    result = result + "#edges nonweighted undirected \n"
    lista = graph.get_edgelist()
    for i in lista:
        edge = str(i[0]) + " " + str(i[1]) + "\n"
        result = result + edge
    with codecs.open(path , "w" , "utf-8" , errors='replace') as temp:
        temp.write(result)


def read_result_file(path):
    #path = 'extras/acc_results.txt'
    index = 0
    result = dict()
    file = open(path)
    for line in file.readlines():
        line = line.rstrip('\n')
        value = float(line)
        result[index] = value
        index+=1
    return result



class CNetwork(object):

    def __init__(self, document, model, percentages, path):
        #server:: Linux-4.4.0-157-generic-x86_64-with-Ubuntu-16.04-xenial
        self.document = document
        self.words = list(set(self.document))
        self.model = model
        self.percentages = percentages
        self.path = path
        plataforma = platform.platform()
        #if plataforma == 'Linux-5.0.0-31-generic-x86_64-with-Ubuntu-18.04-bionic':
        if plataforma.find('Linux')!=-1:
            self.operating_system = 'linux'
        else:
            self.operating_system = 'mac'

    def create_network(self):
        edges = []
        string_bigrams = bigrams(self.document)
        for i in string_bigrams:
            edges.append((i[0], i[1]))
        #words = list(set(self.document))
        network = Graph()
        network.add_vertices(self.words)
        network.add_edges(edges)
        network.simplify()
        print('Nodes:', len(self.words), '-', 'Edges:', len(network.get_edgelist()))
        return network


    def add_embeddings(self, network):
        network_size = network.vcount()
        actual_edges = network.get_edgelist()
        num_edges = network.ecount()
        maximum_num_edges = int((network_size * (network_size - 1)) / 2)
        remaining_edges = maximum_num_edges-num_edges
        print('Testing available edges:',maximum_num_edges, remaining_edges)
        edges_to_add = []

        for percentage in self.percentages:
            value = int(num_edges * percentage / 100) + 1
            edges_to_add.append(value)
        print(edges_to_add)
        #words = list(set(self.document))
        matrix = []
        for word in self.words:
            embedding = self.model[word]
            matrix.append(embedding)

        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        similarity_matrix[np.triu_indices(network_size)] = -1
        similarity_matrix[similarity_matrix == 1.0] = -1
        largest_indices = get_largest_indices(similarity_matrix, maximum_num_edges)

        max_value = np.max(edges_to_add)
        counter = 0
        index = 0
        new_edges = []
        while counter < max_value:
            x = largest_indices[0][index]
            y = largest_indices[1][index]
            if not network.are_connected(x, y):
                new_edges.append((x, y))
                counter += 1
            index += 1

        networks = []
        for value in edges_to_add:
            edges = []
            edges.extend(actual_edges)
            edges.extend(new_edges[0:value])
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(edges)
            networks.append(new_network)
        return networks

    def create_networks(self):
        network = self.create_network()
        networks = self.add_embeddings(network)
        networks.insert(0, network)

        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        return networks

    def get_weighted_network(self, words):
        matrix = []
        for word in words:
            embedding = self.model[word]
            matrix.append(embedding)
        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        simList = similarity_matrix.tolist()
        return Graph.Weighted_Adjacency(simList, mode="undirected", attr="weight", loops=False)

    def get_alpha(self, k, p_ij):
        try:
            alpha = 1 - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
        except:
            alpha = 1
        return alpha

    def disparity_filter(self, network):
        print('Calculating disparity filter')
        degree = network.degree()
        for vertex in range(network.vcount()):
            k = degree[vertex]
            neighbors = network.neighbors(vertex)
            if k > 1:
                sum_w = network.strength(vertex, weights=network.es['weight'])
                for v in neighbors:
                    w = network.es[network.get_eid(vertex, v)]['weight']
                    p_ij = float(np.absolute(w)) / sum_w
                    alpha_ij = self.get_alpha(k, p_ij)
                    network.es[network.get_eid(vertex, v)]['alpha'] = alpha_ij
            else:
                network.es[network.get_eid(vertex, neighbors[0])]['alpha'] = 0

    def add_filtered_embeddings(self, network):
        print('testing ...')
        weighted_network = self.get_weighted_network(self.words)
        self.disparity_filter(weighted_network)
        extra_edges = []
        all_edges = weighted_network.es
        print('Looking for embedding edges')
        for edge_values in all_edges:
            edge = edge_values.tuple
            if not network.are_connected(edge[0], edge[1]):
                extra_edges.append(edge_values)

        sorted_extra_edges = sorted(extra_edges, key=lambda x: x["alpha"], reverse=True)
        num_edges = len(sorted_extra_edges)
        print('Total edges:', len(weighted_network.get_edgelist()))
        print('Embedding edges:', num_edges)
        top_k_to_remove = []
        for percentage in self.percentages:
            top_k = int(num_edges * percentage / 100) + 1
            top_k_to_remove.append(top_k)
        filtered_networks = []
        for top in top_k_to_remove:
            remove = sorted_extra_edges[0:top]
            edges_to_remove = [(e.source, e.target) for e in remove]
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(weighted_network.get_edgelist())
            new_network.delete_edges(edges_to_remove)
            filtered_networks.append(new_network)
        return filtered_networks

    def create_filtered_networks(self):
        network = self.create_network()
        networks = self.add_filtered_embeddings(network)
        networks.insert(0, network)
        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        return networks

    def shortest_path(self, network, features):
        average_short_path = network.shortest_paths(features)
        result = []
        for path in average_short_path:
            average = float(np.divide(np.sum(path), (len(path) - 1)))
            result.append(average)
        return result

    def symmetry(self, network, features):
        in_network = self.path +'auxiliar_network.xnet'
        output = self.path + 'auxiliar.csv'
        generate_xnet(network, in_network)
        if self.operating_system == 'linux':
            path_command = './extras/concentrics/linux/CVSymmetry -c -M -l 3 ' + in_network + ' ' + output
        else:
            path_command = './extras/concentrics/mac/CVSymmetry -c -M -l 3 ' + in_network + ' ' + output

        os.system(path_command)
        simetrias = pd.read_csv(output, sep='\t', lineterminator='\n')

        back_sym_2 = 'Backbone Accessibility h=2'
        merg_sym_2 = 'Merged Accessibility h=2'
        back_sym_3 = 'Backbone Accessibility h=3'
        merg_sym_3 = 'Merged Accessibility h=3'

        v1 = np.array(simetrias[back_sym_2])
        v2 = np.array(simetrias[merg_sym_2])
        v3 = np.array(simetrias[back_sym_3])
        v4 = np.array(simetrias[merg_sym_3])

        sim_results = [v1, v2, v3, v4]
        keys = ['bSym2', 'mSym2', 'bSym3', 'mSym3']
        final_results = dict()

        for key, values in zip(keys,sim_results):
            valid_syms = []
            for word in features:
                node = network.vs.find(name=word)
                valid_syms.append(values[node.index])
            final_results[key] = valid_syms
        return final_results


    def accessibility(self, network, features, h):
        in_network = self.path + 'auxiliar_network.xnet'
        extra_file = self.path + 'acc_results.txt'
        generate_xnet(network, in_network)
        if self.operating_system == 'linux':
            path_command = './extras/concentrics/linux/CVAccessibility -l ' + str(h) + ' ' + in_network + ' > ' + extra_file
        else:
            path_command = './extras/concentrics/mac/CVAccessibility2 -l ' + str(h) + ' ' + in_network + ' > ' + extra_file

        os.system(path_command)
        accs_values = read_result_file(extra_file)
        result = []
        for word in features:
            node = network.vs.find(name=word)
            result.append(accs_values[node.index])
        return result


    def get_network_measures(self, network, features):
        found_features = []
        for word in features:
            try:
                node = network.vs.find(name=word)
            except:
                node = None
            if node is not None:
                found_features.append(word)

        dgr = network.degree(found_features)
        pr = network.pagerank(found_features)
        btw = network.betweenness(found_features)
        cc = network.transitivity_local_undirected(found_features)
        sp = self.shortest_path(network, found_features)
        #symmetries = self.symmetry(network, found_features)
        #bSym2 = symmetries['bSym2']
        #mSym2 = symmetries['mSym2']
        #bSym3 = symmetries['bSym3']
        #mSym3 = symmetries['mSym3']
        #accs_h2 = self.accessibility(network, found_features, 2)
        #accs_h3 = self.accessibility(network, found_features, 3)


        #measures = [dgr, pr, btw, cc, sp, bSym2, mSym2, bSym3, mSym3, accs_h2, accs_h3]
        #measures = [dgr, pr, btw, cc, sp, bSym2, mSym2, bSym3, mSym3, accs_h2]
        measures = [dgr, pr, btw, cc, sp]
        network_features = []
        #network_features2 = []

        for measure in measures:
            feature = [0.0 for _ in range(len(features))]
            for word, value in zip(found_features, measure):
                feature[features[word]] = value

            network_features.extend(feature)
            #network_features2.append(feature)

        network_features = np.array(network_features)
        #network_features2 = np.array(network_features2)


        network_features[np.isnan(network_features)] = 0
        #network_features2[np.isnan(network_features2)] = 0
        print('Len features:', len(network_features))
        return network_features
