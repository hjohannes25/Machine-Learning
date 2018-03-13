
import copy

import numpy as np

__all__ = ['GraphBaggingClassifier']


class GraphBaggingClassifier:
    def __init__(self, estimator, num_estimators):
        self.estimators = list(copy.copy(estimator) for _ in range(num_estimators))
        self.classes_ = None

    def fit(self, X, y):
        g, edges = X
        with g.temp_add_edges(edges):
            bin_size = len(edges)//len(self.estimators)
            for i, estimator in enumerate(self.estimators):
                mask = list(range(i*bin_size//2, (i + 1)*bin_size//2))
                mask += list(range(len(edges)//2 + i*bin_size//2, len(edges)//2 + (i + 1)*bin_size//2))
                with g.temp_remove_edges(edges[mask]):
                    estimator.fit((g, edges[mask]), y[mask])
        self.classes_ = self.estimators[0].classes_

    def predict_proba(self, edges):
        return np.sum((estimator.predict_proba(edges) for estimator in self.estimators))/len(self.estimators)
