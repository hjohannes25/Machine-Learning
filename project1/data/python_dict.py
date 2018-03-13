
import pickle

from project1.data.utils import Graph, iter_adj_list


def load_train(name):
    '''
    Training file is numbered from 0 to n. Not all nodes in the training file have their own row.
    '''
    g = Graph()
    for node, neighbors in iter_adj_list(name):
        for neighbor in neighbors:
            g.add_edge(node, neighbor)
    g.compute_in_deg_dict()
    return g


def main():
    with open('data/train_python_dict.pickle', 'wb') as sr:
        pickle.dump(load_train('data/train.txt'), sr)

if __name__ == '__main__':
    main()
