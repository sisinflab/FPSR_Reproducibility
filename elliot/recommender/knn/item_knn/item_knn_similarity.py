import pickle

import os
import sys

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, manhattan_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import similaripy as sim
from operator import itemgetter

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

class Similarity(object):
    """
    Simple kNN class
    """

    def __init__(self, data, num_neighbors, similarity, implicit, dataset, save_heatmap, **kwargs):
        self._data = data
        self._ratings = data.train_dict
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._implicit = implicit
        self._alpha = kwargs['alpha']
        self._tversky_alpha = kwargs['tversky_alpha']
        self._tversky_beta= kwargs['tversky_beta']

        self.dataset = dataset
        self.save_heatmap = save_heatmap

        if self._implicit:
            self._URM = self._data.sp_i_train
        else:
            self._URM = self._data.sp_i_train_ratings

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def initialize(self):
        """
        This function initialize the data model
        """

        self.supported_similarities = ["cosine", "dot", ]
        self.supported_dissimilarities = ["euclidean", "manhattan", "haversine",  "chi2", 'cityblock', 'l1', 'l2', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        print(f"\nSupported Similarities: {self.supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {self.supported_dissimilarities}\n")

        # self._item_ratings = {}
        # for u, user_items in self._ratings.items():
        #     for i, v in user_items.items():
        #         self._item_ratings.setdefault(i, {}).update({u: v})

        # self._transactions = self._data.transactions

        # self._similarity_matrix = np.empty((len(self._items), len(self._items)))

        # self.process_similarity(self._similarity)
        if self._similarity == "cosine":
             W_sparse = sim.cosine(self._URM.T, k=self._num_neighbors, format_output='csr')
        elif self._similarity == "asym":
             W_sparse = sim.asymmetric_cosine(self._URM.T, alpha=self._alpha, k=self._num_neighbors, format_output='csr')
        elif self._similarity == "dot":
             W_sparse = sim.dot_product(self._URM.T, k=self._num_neighbors, format_output='csr')
        elif self._similarity == "jaccard":
             W_sparse = sim.jaccard(self._URM.T, k=self._num_neighbors, binary=True, format_output='csr')
        elif self._similarity == "dice":
             W_sparse = sim.dice(self._URM.T, k=self._num_neighbors, binary=True, format_output='csr')
        elif self._similarity == "tversky":
             W_sparse = sim.tversky(self._URM.T, k=self._num_neighbors, alpha=self._tversky_alpha, beta=self._tversky_beta, binary=True, format_output='csr')
        # self._similarity_matrix = normalize(self._similarity_matrix, norm='l1', axis=1)

        ##############
        # data, rows_indices, cols_indptr = [], [], []

        # column_row_index = np.arange(len(self._data.items), dtype=np.int32)

        # for item_idx in range(len(self._data.items)):
        #    cols_indptr.append(len(data))
        #    column_data = self._similarity_matrix[:, item_idx]

        #    non_zero_data = column_data != 0

        #    idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
        #    top_k_idx = idx_sorted[-self._num_neighbors:]

        #    data.extend(column_data[non_zero_data][top_k_idx])
        #    rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        #cols_indptr.append(len(data))

        #W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
        #                             shape=(len(self._data.items), len(self._data.items)), dtype=np.float32).tocsr()
        self._preds = self._URM.dot(W_sparse)
        ##############
        # self.compute_neighbors()

        # del self._similarity_matrix

        if self.save_heatmap:
            # to save similarity_matrix in numpy dense version
            logger.info(f"Creating heatmap")
            sim_matrix = W_sparse.todense()
            logger.info(f"Min: {np.min(sim_matrix)}")
            logger.info(f"Max: {np.max(sim_matrix)}")
            logger.info(f"Mean: {np.mean(sim_matrix)}")
            logger.info(f"Std: {np.std(sim_matrix)}")
            np.save(f'{os.getcwd()}/heatmap/{self.dataset}/itemknn/similarity_matrix.npy', sim_matrix)
            logger.info(f"Created")
            del sim_matrix

    # def compute_neighbors(self):
    #     self._neighbors = {}
    #     for x in range(self._similarity_matrix.shape[0]):
    #         arr = np.concatenate((self._similarity_matrix[0:x, x], [-np.inf], self._similarity_matrix[x, x+1:]))
    #         top_indices = np.argpartition(arr, -self._num_neighbors)[-self._num_neighbors:]
    #         arr = arr[top_indices]
    #         self._neighbors[self._private_items[x]] = {self._private_items[i]: arr[p] for p, i in enumerate(top_indices)}
    #
    # def get_item_neighbors(self, item):
    #     return self._neighbors.get(item, {})

    def process_similarity(self, similarity):
        if similarity == "cosine":
            # x, y = np.triu_indices(self._similarity_matrix.shape[0], k=1)
            # self._similarity_matrix[x, y] = cosine_similarity(self._data.sp_i_train_ratings.T)[x, y]
            self._similarity_matrix = cosine_similarity(self._URM.T)
        elif similarity == "dot":
            self._similarity_matrix = (self._URM.T @ self._URM).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM.T)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._URM.T)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._URM.T)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._URM.T)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:

            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T.toarray(), metric=similarity)))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\nTry with implementation: aiolli")

    # def process_cosine(self):
    #     x, y = np.triu_indices(self._similarity_matrix.shape[0], k=1)
    #     self._similarity_matrix[x, y] = cosine_similarity(self._data.sp_i_train_ratings.T)[x, y]
    #     # g = np.vectorize(self.compute_cosine)
    #     # g(x,y)
    #     # for item_row in range(self._similarity_matrix.shape[0]):
    #     #     for item_col in range(item_row + 1, self._similarity_matrix.shape[1]):
    #     #         self._similarity_matrix[item_row, item_col] = self.compute_cosine(
    #     #             self._item_ratings.get(self._private_items[item_row],{}), self._item_ratings.get(self._private_items[item_col], {}))
    #
    # def compute_cosine(self, i_index, j_index):
    #     i_dict = self._item_ratings.get(self._private_items[i_index],{})
    #     j_dict = self._item_ratings.get(self._private_items[j_index],{})
    #     union_keyset = set().union(*[i_dict, j_dict])
    #     i: np.ndarray = np.array([[i_dict.get(x, 0) for x in union_keyset]])
    #     j: np.ndarray = np.array([[j_dict.get(x, 0) for x in union_keyset]])
    #     self._similarity_matrix[i_index, j_index] = cosine_similarity(i, j)[0, 0]
    #
    # def get_transactions(self):
    #     return self._transactions

    # def get_user_recs(self, u, mask, k):
    #     user_items = self._ratings[u].keys()
    #     user_mask = mask[self._data.public_users[u]]
    #     predictions = {i: self.score_item(self.get_item_neighbors(i), user_items) for i in self._data.items if
    #                    user_mask[self._data.public_items[i]]}
    #
    #     indices, values = zip(*predictions.items())
    #     indices = np.array(indices)
    #     values = np.array(values)
    #     local_k = min(k, len(values))
    #     partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
    #     real_values = values[partially_ordered_preds_indices]
    #     real_indices = indices[partially_ordered_preds_indices]
    #     local_top_k = real_values.argsort()[::-1]
    #     return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_user_recs_batch(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        users_recs = np.where(mask[u_index, :], self._preds[u_index, :].toarray(), -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]
    # @staticmethod
    # def score_item(neighs, user_items):
    #     num = sum([v for k, v in neighs.items() if k in user_items])
    #     den = sum(np.power(list(neighs.values()), 1))
    #     return num/den if den != 0 else 0

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
