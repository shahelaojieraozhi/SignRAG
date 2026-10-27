import torch
import numpy as np
import torch.nn as nn
import pdb
import math
import copy


class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1, **kwargs):
        if 'max_hop' in kwargs:
            max_hop = kwargs['max_hop']
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout) # set up the connections between nodes
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop) # compute the hop distance between nodes: 0 = self, 1 = adjacent node, 2 = neighbor of a neighbor, ...
        self.get_adjacency(strategy) # compute the adjacency matrix

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'custom_hand21':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [0, 17],
                [17, 18],
                [18, 19],
                [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'custom_hand':
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 6],
                [0, 8],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'custom_body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [3, 5],
                [5, 7],
                [4, 6],
                [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'custom_mouth_8':
            self.num_node = 8
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(self.num_node - 1)] + [
                [self.num_node - 1, 0]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'custom_face':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(16)] + [
                [17, i] for i in range(17)
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 8

    def get_adjacency(self, strategy): # compute the adjacency matrix from the node connections
        """
        Get adjacency matrix
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation) # compute the valid hop distance range
        adjacency = np.zeros((self.num_node, self.num_node)) # initialize the adjacency matrix
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency) # normalize the adjacency matrix

        if strategy == 'uniform': # all nodes share a single adjacency matrix
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance': # split into multiple adjacency matrices by hop distance
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial': # split into multiple adjacency matrices by distance from the center node
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD