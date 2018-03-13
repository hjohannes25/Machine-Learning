
import pickle

from graph_tool.all import Graph
from project1.data.utils import iter_adj_list


def load_train(name):
    '''
    Training file is numbered from 0 to n. Not all nodes in the training file have their own row.
    '''
    g = Graph()
    node_ids = set()
    n = -1
    for n, (node_id, neighbor_ids) in enumerate(iter_adj_list(name)):
        node_ids.add(node_id)
        node_ids.update(neighbor_ids)
    n += 1
    g.add_vertex(len(node_ids))
    for i, (node_id, neighbor_ids) in enumerate(iter_adj_list(name)):
        print('adding edge for vertex {}/{}'.format(i + 1, n))
        for neighbor_id in neighbor_ids:
            g.add_edge(node_id, neighbor_id)
    return g


def main():
    with open('data/train_graph_tool.pickle', 'wb') as sr:
        pickle.dump(load_train('data/train.txt'), sr)


if __name__ == '__main__':
    main()
