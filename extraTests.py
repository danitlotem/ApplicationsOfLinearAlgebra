import math
import random
import networkx as nx
import numpy as np
from numpy.linalg import norm
from networkx.algorithms import bipartite

def circle(n):
    g = nx.DiGraph()

    for i in range(n):
        g.add_node(i)
    i = 0
    for i in range(0, n-1, 1):
        g.add_edge(i, i+1)

    g.add_edge(n-1, 0)

    print("build finished")
    return g


def bipartite(n,q):
    g = nx.DiGraph()

    for i in range(n):
        g.add_node(i)
    i = 0
    for i in range(0, int(n/2), 1):
        for j in (int(n/2), n-1, 1):
            if np.random.choice(np.arange(1, 3), p=[q, 1 - q]) == 1:
                g.add_edge(i, j)

    i = 0

    for i in range(0, int(n/2), 1):
        for j in (int(n/2), n-1, 1):
            if np.random.choice(np.arange(1, 3), p=[q, 1 - q]) == 1:
                g.add_edge(j, i)

    print("build finished")
    return g

def calculate_norma_of_vector(vector1, vector2):
    new_vector = vector1 - vector2
    norm_vector = norm(new_vector)

    return norm_vector


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

    d = np.zeros(n)
    curr_node = random.randint(0, n - 1)

    for i in range(t):
        for j in range(N):
            neighbors_list = list(graph.neighbors(curr_node))

            if np.random.choice(np.arange(1, 3), p=[prob, 1 - prob]) == 1:
                curr_node = random.choice(list(graph.nodes))
            else:
                if len(neighbors_list) == 0:
                    curr_node = curr_node
                else:
                    curr_node = random.choice(neighbors_list)

        d[curr_node-1] = d[curr_node-1] + 1
        curr_node = random.randint(0, n - 1)

    normalized_d = normalize_d_arr(d, t)

    return normalized_d


def test():
    n = int(math.pow(2, 12))
    #graph_1 = bipartite(n,1/2)
    graph_1 = circle(n)
    # graph_2 = create_graph_family_one(n, 1/(int(math.pow(2, 9))))
    #graph_3 = create_graph_family_one(n, 1 / (int(math.pow(2, 4))))
    #nx.draw_networkx(graph_1, pos=nx.drawing.layout.bipartite_layout(graph_1, lst1), width=2)


    p = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    epsilon = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    #epsilon=[1/64,1/128]

    for epsilon_elem in epsilon:
        for p_elem in p:
            N = 32
            print("number of edges:" + str(graph_1.number_of_edges()))
            t_res_graph = page_rank_first_stop_condition(graph_1, N, p_elem, epsilon_elem, n)

            print("p:" + str(p_elem) + ", N:" + str(N) + ", epsilon:" + str(epsilon_elem) + ", t:" + str(t_res_graph))
        print("\n")


if __name__ == '__main__':
    test()


