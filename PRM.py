# -----------------------------
# Python3.8
# encoding    : utf-8
# @Time       : 2022/4/29 10:36
# @Software   : PyCharm
# @Description:PRM path planning
# -----------------------------
import pickle
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sympy
import tqdm
from sklearn.neighbors import KDTree


GraphType = Dict[Tuple, List[Tuple]]

root = Path.cwd()
im_path = root / 'map.png'

Inf = np.inf
map = mpimg.imread(im_path)
map = map[:, :, 0]

start = (0, 0)
goal = map.shape


def rc22imgxy(rcxy):
    cx = rcxy[1]  # matrix column to imgx
    ry = rcxy[0]  # matrix row to imgy
    return cx, ry


class Node:
    def __init__(self, xy: Tuple = None, father_node=None, cost=None):
        if cost is None:
            cost = [0, 0, 0]
        self.xy = xy
        self.father_node = father_node
        self.gn = cost[0]  # fn for startpoint to current node
        self.hn = cost[1]  # fn for current node to goalpoint
        self.fn = cost[2]  # total cost


class AStar:
    def __init__(self, map, graph: GraphType, startpoint: Tuple, goalpoint: Tuple):
        self.map = map
        self.graph = graph
        self.startpoint = startpoint
        self.goalpoint = goalpoint
        self.openset = {}
        self.closeset = {}
        self.pathnode = []

    def planner(self):
        st = time.time()
        startpoint = Node(xy=self.startpoint)
        startpoint.gn, startpoint.hn, startpoint.fn = self.__cal_cost(startpoint)
        self.openset[startpoint.xy] = startpoint
        goalpoint = Node(xy=self.goalpoint)
        i = 0
        # 动态显示寻路过程
        # plt.imshow(self.map,cmap='gray')
        # for j in self.graph.keys():
        #     for k in self.graph[j]:
        #         plt.plot([j[0],k[0]],[j[1],k[1]],color='green', linestyle='-', linewidth=0.5)
        # plt.ion()
        while self.openset:
            i = i + 1
            print(i)
            current_node = sorted(self.openset.items(), key=lambda x: x[1].fn)[0][0]
            # plt.scatter(current_node[0],current_node[1], color='red')
            # plt.pause(0.2)
            self.closeset[current_node] = self.openset[current_node]
            self.openset.pop(current_node)
            if current_node == goalpoint.xy:
                while True:
                    if current_node == self.startpoint:
                        self.pathnode.append(current_node)
                        self.pathnode = list(reversed(self.pathnode))
                        break
                    self.pathnode.append(current_node)
                    current_node = self.closeset[current_node].father_node
                break
            else:
                for nn in self.graph[current_node]:
                    if nn in self.closeset.keys():
                        continue
                    elif nn not in self.openset.keys():
                        neighbor_node = Node(xy=nn, father_node=current_node)
                        neighbor_node.gn, neighbor_node.hn, neighbor_node.fn = self.__cal_cost(neighbor_node)
                        self.openset[nn] = neighbor_node
                    elif nn in self.openset.keys():
                        neighbor_node = self.openset[nn]
                        gn = self.closeset[current_node].gn + self.__distance(neighbor_node.xy,
                                                                              self.closeset[current_node].xy)
                        if gn <= neighbor_node.gn:
                            neighbor_node.father_node = current_node
                            neighbor_node.gn, neighbor_node.hn, neighbor_node.fn = self.__cal_cost(neighbor_node)
                            self.openset[nn] = neighbor_node

        # plt.ioff()
        # plt.show()
        et = time.time()
        print('solution time', et - st)
        self.plot_path()

    def __cal_cost(self, node: Node):
        if node.father_node is None:
            gn = 0
        else:
            father_node = self.closeset[node.father_node]
            gn = father_node.gn + self.__distance(father_node.xy, node.xy)
        hn = self.__distance(node.xy, self.goalpoint)
        fn = gn + hn
        return gn, hn, fn

    @staticmethod
    def __distance(xy1: Tuple, xy2: Tuple):
        xy1 = np.array(xy1)
        xy2 = np.array(xy2)
        distance = np.sqrt(np.sum((xy1 - xy2) ** 2))
        return distance

    def plot_path(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(self.map, cmap='gray')
        for i in self.graph.keys():
            for j in self.graph[i]:
                ax.plot([i[0], j[0]], [i[1], j[1]],
                        color='gray', linestyle='-', linewidth=0.5,
                        marker='o', markersize=1, markeredgecolor='gray', markerfacecolor='blue')
        for i in range(len(self.pathnode) - 1):
            ax.plot([self.pathnode[i][0], self.pathnode[i + 1][0]], [self.pathnode[i][1], self.pathnode[i + 1][1]],
                    color='black', linestyle='-', linewidth=0.5,
                    marker='o', markersize=3, markeredgecolor='red', markerfacecolor='red')
        ax.axis('off')
        fig.savefig('./path.png', bbox_inches='tight', dpi=300)


class PRM:
    def __init__(self, nnodes, nneighbors, map, startpoint=None, goalpoint=None):
        self.nnodes = nnodes  # number of nodes to put in the roadmap
        self.nneighbors = nneighbors  # number of nearest neighbors to examine for each configuration
        self.graph: GraphType = {}  # roadmap G=(V,E)
        self.nodes = []
        self.map = map
        self.startpoint = startpoint
        self.goalpoint = goalpoint
        self.pathnode = None
        self.kdtree = None

    def planner(self):
        self.nodes, self.graph = self.construct_roadmap()

        startpoint_edge = self.get_edges(self.startpoint, self.nodes, self.kdtree)
        goalpoint_edge = self.get_edges(self.goalpoint, self.nodes, self.kdtree)
        if len(startpoint_edge) + len(goalpoint_edge) == 0:
            print('the number of nodes to put in the roadmap can not enough to plan a path!')
        else:
            self.graph[tuple(self.startpoint)] = [startpoint_edge[0]]
            self.graph[tuple(self.goalpoint)] = [goalpoint_edge[0]]
            self.graph[self.graph[self.goalpoint][0]].append(self.goalpoint)

        with open('graph.pkl', 'wb') as f:
            pickle.dump(self.graph, f)

        self.plot_roadmap(self.map, self.graph)
        astar = AStar(self.map, self.graph, self.startpoint, self.goalpoint)
        astar.planner()
        self.pathnode = astar.pathnode
        print(self.pathnode)



    def construct_roadmap(self):
        nodes = self.get_nodes()
        self.kdtree = KDTree(nodes, metric='euclidean', leaf_size=2)
        graph = {}
        with tqdm.tqdm(nodes) as pbar:
            for qnode in pbar:
                neighbor_nodes_free = self.get_edges(qnode, nodes, self.kdtree)
                graph[qnode] = neighbor_nodes_free
                pbar.set_description(f'getting the edges of the {nodes.index(qnode)+1}th node')
        return nodes, graph

    def get_nodes(self):
        nodes = []
        h, w = self.map.shape  # 图像坐标系
        collision_free = np.argwhere(self.map == 1).tolist()
        collision_free = [rc22imgxy(rc) for rc in collision_free]
        while len(nodes) < self.nnodes:
            x, y = random.randint(0, w), random.randint(0, h)
            if (x, y) not in nodes and (x, y) in collision_free:
                nodes.append((x, y))
        return nodes

    def get_edges(self, qnode: Tuple, nodes: List[Tuple], kdtree=None):
        # neighbor_nodes, dist = self.get_nearest_neighbors(qnode, nodes)
        neighbor_nodes_index = self.get_nearest_neighbors_kdtree(qnode, kdtree)
        neighbor_nodes = [nodes[i] for i in neighbor_nodes_index]
        neighbor_nodes_free = []
        for i in neighbor_nodes:
            if self.check_collision_existence(qnode, i):
                neighbor_nodes_free.append(i)
        return neighbor_nodes_free

    def get_nearest_neighbors(self, qnode, nodes) -> (List[Tuple], List):
        qnode = np.array(qnode)
        nodes = np.array(nodes)
        dist = np.sqrt(np.sum((qnode - nodes) ** 2, axis=1))
        arg = np.argsort(dist)
        neighbor_nodes = nodes[arg[1:self.nneighbors + 1], :].tolist()
        neighbor_nodes = [tuple(nn) for nn in neighbor_nodes]
        dist = dist[arg[1:self.nneighbors + 1]].tolist()
        return neighbor_nodes, dist

    @staticmethod
    def get_nearest_neighbors_kdtree(qnode, kdtree):
        neighbors_index = kdtree.query(np.array([qnode]), return_distance=False, k=6)
        return neighbors_index[0, 1:]

    def check_collision_existence(self, node1, node2):
        """check whether collision existence on the path of node1 to node2"""
        collision = np.argwhere(self.map == 0).tolist()
        collision = [rc22imgxy(rc) for rc in collision]
        A, B, C = self.get_line_equation(node1, node2)
        y = sympy.Symbol('y')
        row = sorted([node1[0], node2[0]])
        node1_2 = [(x, round(sympy.solve(A * x + B * y + C, y)[0])) for x in range(row[0], row[1])]
        for n in node1_2:
            if n in collision:
                return False
        return True

    @staticmethod
    def get_line_equation(node1: List, node2: List):
        # Ax+By+C=0
        A = node2[1] - node1[1]
        B = node1[0] - node2[0]
        C = node2[0] * node1[1] - node1[0] * node2[1]
        return A, B, C

    def plot_roadmap(self, map, graph: Dict):
        fig = plt.Figure()
        ax = fig.add_subplot()
        ax.imshow(map, cmap='gray')
        for node, neighbor_nodes in graph.items():
            imgx_node, imgy_node = node[0], node[1]
            for nn in neighbor_nodes:
                imgx_nn, imgy_nn = nn[0], nn[1]
                if node in [self.startpoint, self.goalpoint]:
                    ax.plot([imgx_node, imgx_nn], [imgy_node, imgy_nn],
                            color='green', linestyle='-', linewidth=0.5)
                    ax.scatter(imgx_node, imgy_node, c='red')
                else:
                    ax.plot([imgx_node, imgx_nn], [imgy_node, imgy_nn],
                            color='green', linestyle='-', linewidth=0.5,
                            marker='o', markersize=1, markeredgecolor='blue', markerfacecolor='blue')
                ax.axis('off')

        fig.savefig('graph.png', dpi=300, bbox_inches='tight')
        plot_node(self.map, self.graph)


def plot_node(map, graph):
    plt.imshow(map, cmap='gray')
    for i in graph.keys():
        plt.scatter(i[0], i[1], c='red')
    plt.axis('off')
    plt.savefig('nodes.png', dpi=300, bbox_inches='tight')


prm = PRM(100, 5, map, start, goal)
prm.planner()


print('Finished!\U0001F604')
