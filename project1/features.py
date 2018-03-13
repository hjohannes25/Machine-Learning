
from queue import deque

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Degrees', 'CommonNeighbors', 'AdamicAdar', 'Katz']


def printFeature(label,res):
    with open('data/feat-'+label+'.txt','w') as f:
        for l in res:
            print(l,file=f)
    return res


class BaseGraphEstimator(BaseEstimator):
    def fit(self, edges, y=None):
        return self


class Degrees(BaseGraphEstimator):
    @staticmethod
    def transform(X):
        g, edges = X
        res = []
        for u, v in edges:
            res.append(np.array([
                len(g.in_dict[u]), len(g.out_dict[u]),
                len(g.in_dict[v]), len(g.out_dict[v]),
            ]))
        printFeature('degree',np.vstack(res))
        return  printFeature('logdegree',np.log(np.vstack(res) + 1))


class CommonNeighbors(BaseGraphEstimator):
    @staticmethod
    def transform(X):
        g, edges = X
        res = []
        for u, v in edges:
            u_in = set(g.in_dict[u])
            u_out = set(g.out_dict[u])
            v_in = set(g.in_dict[v])
            v_out = set(g.out_dict[v])
            res.append(np.array([len(u_in & v_in), len(u_in & v_out), len(u_out & v_in), len(u_out & v_out)]))
        return printFeature('common-neighbors',np.vstack(res))


class AdamicAdar(BaseGraphEstimator):
    @staticmethod
    def transform(X):
        g, edges = X
        res = []
        for u, v in edges:
            u_in = set(g.in_dict[u])
            u_out = set(g.out_dict[u])
            v_in = set(g.in_dict[v])
            v_out = set(g.out_dict[v])
            res.append(np.array([
                sum(1/np.log(len(g.out_dict[z])) for z in u_in & v_in),
                sum(1/np.log(len(g.in_dict[z]) + len(g.out_dict[z])) for z in u_in & v_out),
                sum(1/np.log(len(g.in_dict[z]) + len(g.out_dict[z])) for z in u_out & v_in),
                sum(1/np.log(len(g.in_dict[z])) for z in u_out & v_out),
            ]))
        return printFeature('adamic-adar',np.vstack(res))


class Katz(BaseGraphEstimator):
    '''
    Does not search the entire graph due to the high computational cost of even partial matrix inversion.
    '''

    def __init__(self, depth, beta):
        self.depth = depth
        self.beta = beta

    def transform(self, X):
        g, edges = X
        res = []
        for (u, v) in edges:
            # bfs search
            score = 0
            q = deque([u])
            cur_depth = 0
            while q and cur_depth < self.depth:
                cur_depth += 1
                node = q.popleft()
                for neighbor in g.out_dict[node]:
                    if neighbor == v:
                        score += self.beta**cur_depth
                    q.append(neighbor)
            res.append(score)
        return printFeature('katz',np.vstack(res))
