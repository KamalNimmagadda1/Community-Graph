import sys
import time
import random
from itertools import combinations
from pyspark import SparkContext


class GraphFrame():

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.vertex_weight = dict([(vertex, 1) for vertex in self.vertices])

        self.edges = edges
        self.__init_adjacency_matrix(edges)

        self.betweenness_dict = dict()
        self.betweenness_list = None

        self.valid_communities = None

    def __init_adjacency_matrix(self, edges):
        self.org_edges = edges

        visited = set()
        self.c = 0
        for i, j_list in edges.items():
            for j in j_list:
                edg = (i, j) if i < j else (j, i)
                if edg not in visited:
                    visited.add(edg)
                    self.c += 1
        self.adjacency_matrix = visited

    def get_tree(self, root):
        tree = dict()
        tree[root] = (0, list())

        visited = set()

        nv = [root]

        while len(nv):
            parent = nv.pop(0)
            visited.add(parent)
            for child in self.edges[parent]:
                if child not in visited:
                    visited.add(child)
                    tree[child] = (tree[parent][0] + 1, [parent])
                    nv.append(child)
                elif tree[parent][0] + 1 == tree[child][0]:
                    tree[child][1].append(parent)

        tree = {k1: v1 for k1, v1 in sorted(tree.items(), key= lambda x: -x[1][0])}

        return tree

    def get_num_path(self, tree):
        tree_level = dict()
        shortest_path = dict()

        for child, parents in tree.items():
            if parents[0] not in tree_level.keys():
                tree_level[parents[0]] = []
            tree_level[parents[0]].append((child, parents[1]))

        for lvl in range(len(tree_level.keys())):
            for (chld, prnt_lst) in tree_level[lvl]:
                if len(prnt_lst) > 0:
                    shortest_path[chld] = sum([shortest_path[prnt] for prnt in prnt_lst])

                else:
                    shortest_path[chld] = 1
        return shortest_path

    def traverse_tree(self, tree):
        weights = self.vertex_weight.copy()
        shortest_path = self.get_num_path(tree)
        weight_update = dict()

        for node, par in tree.items():
            if len(par[1]) > 0:
                denum = sum([shortest_path[pr] for pr in par[1]])
                for pr in par[1]:
                    edg = (node, pr) if node < pr else (pr, node)
                    contribution = float(float(weights[node]) * int(shortest_path[pr]) / denum)
                    weight_update[edg] = contribution
                    ow = weights[pr]
                    weights[pr] = float(ow + contribution)

        return weight_update

    def getBetweenness(self):
        self.betweenness_dict = dict()
        for vertex in self.vertices:
            tree = self.get_tree(root=vertex)
            temp_btw = self.traverse_tree(tree)

            for k, v in temp_btw.items():
                if k in self.betweenness_dict.keys():
                    ow1 = self.betweenness_dict[k]
                    self.betweenness_dict[k] = float(ow1 + v)
                else:
                    self.betweenness_dict[k] = v

        self.betweenness_dict = dict(map(lambda x: (x[0], float(x[1] / 2)), self.betweenness_dict.items()))

        self.betweenness_list = sorted(self.betweenness_dict.items(), key=lambda x: (-x[1], x[0][0]))

        return self.betweenness_list

    def cut_btw_edge(self):
        edg_pair = self.betweenness_list[0][0]

        if self.edges[edg_pair[0]] is not None:
            try:
                self.edges[edg_pair[0]].remove(edg_pair[1])
            except ValueError:
                pass

        if self.edges[edg_pair[1]] is not None:
            try:
                self.edges[edg_pair[1]].remove(edg_pair[0])
            except ValueError:
                pass

    def detectCommunities(self):
        communities = list()
        tmp_set = set()
        visited = set()

        root = self.vertices[random.randint(0, len(self.vertices) - 1)]
        tmp_set.add(root)
        nv = [root]
        while len(visited) != len(self.vertices):
            while len(nv) > 0:
                parent = nv.pop(0)
                tmp_set.add(parent)
                visited.add(parent)
                for child in self.edges[parent]:
                    if child not in visited:
                        tmp_set.add(child)
                        nv.append(child)
                        visited.add(child)

            communities.append(sorted(tmp_set))
            tmp_set = set()
            if len(self.vertices) > len(visited):
                dif_vertices = set(self.vertices).difference(visited).pop()
                nv.append(dif_vertices)
        return communities

    def computeModularity(self):
        communities = self.detectCommunities()

        t = 0
        for community in communities:
            for pair in combinations(list(community), 2):
                edg = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                c_i = len(self.edges[pair[0]])
                c_j = len(self.edges[pair[1]])
                p = 1 if edg in self.adjacency_matrix else 0
                t += float(p - (c_i * c_j / (2 * self.c)))

        return communities, float(t / (2 * self.c))

    def getCommunities(self):
        max_modularity = float("-inf")

        if len(self.betweenness_list) > 0:
            self.cut_btw_edge()
            self.valid_communities, max_modularity = self.computeModularity()
            self.betweenness_list = self.getBetweenness()

        while True:
            self.cut_btw_edge()
            communities, new_modularity = self.computeModularity()
            self.betweenness_list = self.getBetweenness()
            if new_modularity < max_modularity:
                break
            else:
                self.valid_communities = communities
                max_modularity = new_modularity

        return sorted(self.valid_communities, key=lambda x: (len(x), x[0], x[1]))


if __name__=="__main__":
    st = time.time()
    sc = SparkContext("local[*]")
    sc.setLogLevel("ERROR")

    filter_threshold = int(sys.argv[1])
    betweenness_file = sys.argv[3]
    communities_file = sys.argv[4]

    users = sc.textFile(sys.argv[2]).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id") \
        .groupByKey().map(lambda x: (x[0], list(set(x[1])))).collectAsMap()

    user_pairs = sorted(combinations(list(users.keys()), 2))

    edge_pairs = []
    vertex = set()
    for user in user_pairs:
        if len(set(users.get(user[0])) & set(users.get(user[1]))) >= filter_threshold:
            edge_pairs.append((user[0], user[1]))
            edge_pairs.append((user[1], user[0]))
            vertex.add(user[0])
            vertex.add(user[1])

    vertices = sc.parallelize(sorted(list(vertex))).collect()
    edges = sc.parallelize(edge_pairs).groupByKey().map(lambda x: (x[0], sorted(list(set(x[1]))))).collectAsMap()

    com_graph = GraphFrame(vertices, edges)
    res_btw = com_graph.getBetweenness()
    res_com = com_graph.getCommunities()

    with open(betweenness_file, "w") as f1:
        for i in res_btw:
            f1.write(str(i)[1:-1] + "\n")
        f1.close()

    with open(communities_file, "w") as f2:
        for j in res_com:
            f2.write(str(j).strip("[]") + "\n")
        f2.close()

    total_time = time.time() - st
    print("Duration:", total_time)