import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import pickle

def create_graph_family_one(n, q):
    # q = round(random.uniform(0, 1), 2) #probability to add edge to graph
    # g = nx.erdos_renyi_graph(n, q)  # create random graph

    g = nx.Graph()  # build empty graph

    for i in range(n):
        g.add_node(i)

    for i in range(n):
        for j in range(n):
            if np.random.choice(np.arange(1, 3), p=[q, 1 - q]) == 1:
                g.add_edge(i, j)

    return g


def create_graph_family_two(n, q):
    g = nx.Graph()

    for i in range(1, n + 1):
        g.add_node(i)

    nodes_and_probability = {}
    for i in range(1, n + 1):
        nodes_and_probability[i] = 0

    for i in range(1, n + 1):
        # q = round(random.uniform(0, 1), 2)
        q = round(1 / i, 2)
        nodes_and_probability[i] = q

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if np.random.choice(np.arange(1, 3), p=[nodes_and_probability[j], 1 - nodes_and_probability[j]]) == 1:
                g.add_edge(i, j)

    return g


def calculate_norma_of_vector(vector1, vector2):
    new_vector = vector1 - vector2
    norm_vector = norm(new_vector)

    return norm_vector


# normalize page rank arr (d array)
def normalize_d_arr(d, t):
    for i in range(len(d)):
        d[i] = d[i] / t
    return d


def page_rank_first_stop_condition(graph, N, prob, epsilon, n):
    i = 1

    while True:
        v = page_rank(graph, int(math.pow(2, i)), N, prob, n)
        u = page_rank(graph, int(math.pow(2, i - 1)), N, prob, n)
        i += 1

        if calculate_norma_of_vector(v, u) < epsilon:
            break;

    return int(math.pow(2, i - 1))


def page_rank(graph, t, N, prob, n):
    # numOfNodesInGraph= len(graph.nodes)

    # initialize d arr (page rank arr)
    d = np.zeros(n + 1)

    curr_node = random.randint(1, n)  # choosing random node for start
    # d[curr_node] = d[curr_node] + 1

    for i in range(t):
        for j in range(N):
            neighbors_list = list(graph.neighbors(curr_node))

            if np.random.choice(np.arange(1, 3), p=[prob, 1 - prob]) == 1:
                curr_node = random.choice(list(graph.nodes))
            else:
                if len(neighbors_list) == 0:  # if there are no neighbours, stay in current node
                    curr_node = curr_node
                else:  # if there are neighbours, we choose random node with probability of p or random neighbour with probability of 1-p
                    curr_node = random.choice(neighbors_list)

        d[curr_node] = d[curr_node] + 1
        curr_node = random.randint(1, n)

    # return normalized page rank arr
    normalized_d = normalize_d_arr(d, t)

    return normalized_d


def test():
    n = int(math.pow(2, 12))
    # n = int(math.pow(12, 2))
    graph_1 = pickle.load(open(r'C:/Users/danit/Lini/family2graph1realOne.txt', 'rb'))
    #graph_1 = create_graph_family_two(n, 1 / (int(math.pow(2, 12))))
    # graph_2 = create_graph_family_two(n, 1/(int(math.pow(2, 9))))
    # graph_3 = create_graph_family_two(n, 1/(int(math.pow(2, 4))))

    #p = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    p=[1]
    epsilon = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

    for epsilon_elem in epsilon:
        for p_elem in p:
            N = int(1 / p_elem)
            print("number of edges:" + str(graph_1.number_of_edges()))
            t_res_graph1 = page_rank_first_stop_condition(graph_1, N, p_elem, epsilon_elem, n)
            print("p:" + str(p_elem) + ", N:" + str(N) + ", epsilon:" + str(epsilon_elem) + ", t:" + str(t_res_graph1))
            if p_elem==1/32 and epsilon_elem==1/32 :
                print("im here")
        print("\n")


if __name__ == '__main__':
    test()


