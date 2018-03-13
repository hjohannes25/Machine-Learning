
from contextlib import contextmanager
from collections import defaultdict

__all__ = ['Graph', 'iter_adj_list']


def iter_adj_list(name):
    with open(name) as sr:
        for line in sr:
            node_ids = list(map(int, line.split()))
            yield node_ids[0], node_ids[1:]


class Graph:
    def __init__(self):
        self.out_dict = {}
        self.in_dict = {}
        self.num_vertices = 0
        self.num_edges = 0
        self.in_deg_dict = defaultdict(list)
        self.in_deg_dict_computed = 0

    def add_edge(self, u, v):
        if u not in self.out_dict:
            self.num_vertices += 1
            self.out_dict[u] = []
            self.in_dict[u] = []
        if v not in self.out_dict:
            self.num_vertices += 1
            self.out_dict[v] = []
            self.in_dict[v] = []
        self.out_dict[u].append(v)
        self.in_dict[v].append(u)
        self.num_edges += 1

    def remove_edge(self, u, v):
        self.out_dict[u].remove(v)
        self.in_dict[v].remove(u)
        self.num_edges -= 1

    def compute_in_deg_dict(self):
        if self.in_deg_dict_computed:
            return
        for k in self.in_dict.keys():
            self.in_deg_dict[len(self.in_dict[k])].append(k)
        self.in_deg_dict_computed = 1

    @contextmanager
    def temp_add_edges(self, edges):
        for u, v in edges:
            self.add_edge(u, v)
        yield
        for u, v in edges:
            self.remove_edge(u, v)

    @contextmanager
    def temp_remove_edges(self, edges):
        for u, v in edges:
            self.remove_edge(u, v)
        yield
        for u, v in edges:
            self.add_edge(u, v)
