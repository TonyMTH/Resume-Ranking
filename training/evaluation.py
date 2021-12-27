import pickle
from time import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


class Evaluate:
    def __init__(self):
        pass

    def dcg(self, relevances, rank=10):
        """
        Discounted cumulative gain at rank (DCG)
        :param relevances:
        :param rank: int
        :return: float
        """
        relevances = np.asarray(relevances)[:rank]
        n_relevances = len(relevances)
        if n_relevances == 0:
            return 0.

        discounts = np.log2(np.arange(n_relevances) + 2)
        return np.sum(relevances / discounts)

    def ndcg(slef, relevances, rank=10):
        """
        Normalized discounted cumulative gain (NDGC)
        :param relevances:
        :param rank: int
        :return: float
        """
        best_dcg = slef.dcg(sorted(relevances, reverse=True), rank)
        if best_dcg == 0:
            return 0.
        return slef.dcg(relevances, rank) / best_dcg

    def mean_ndcg(self, y_true, y_pred, query_ids, rank=10):
        """
        Mean Normalized discounted cumulative gain (NDGC) for evaluating ranking
        :param y_true: array
        :param y_pred: array
        :param query_ids: array
        :param rank: int
        :return: float
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        query_ids = np.asarray(query_ids)
        # assume query_ids are sorted
        ndcg_scores = []
        previous_qid = query_ids[0]
        previous_loc = 0
        for loc, qid in enumerate(query_ids):
            if previous_qid != qid:
                chunk = slice(previous_loc, loc)
                ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
                ndcg_scores.append(self.ndcg(ranked_relevances, rank=rank))
                previous_loc = loc
            previous_qid = qid

        chunk = slice(previous_loc, loc + 1)
        ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
        ndcg_scores.append(self.ndcg(ranked_relevances, rank=rank))
        return np.mean(ndcg_scores)

    def print_evaluation(self, model, X, y, qid, rank=4):
        tic = time()
        y_predicted = model.predict(X)
        prediction_time = time() - tic
        print("Prediction time: {:.3f}s".format(prediction_time))
        print("NDCG score: {:.3f}".format(
            self.mean_ndcg(y, y_predicted, qid, rank=rank)))
        print("R2 score: {:.3f}".format(r2_score(y, y_predicted)))

    def save_model(self, model, pth):
        with open(pth, "wb") as f:
            pickle.dump(model, f)


class ClassificationRanker(RegressorMixin):

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.base_estimator)
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)
        self.estimator_.fit(X, y)

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        probas = self.estimator_.predict_proba(X_scaled)
        return self.proba_to_relevance(probas)

    def proba_to_relevance(self, probas):
        """MCRank-like reduction of classification proba to DCG predictions"""
        rel = np.zeros(probas.shape[0], dtype=np.float32)
        for i in range(probas.shape[1]):
            rel += i * probas[:, i]
        return rel
