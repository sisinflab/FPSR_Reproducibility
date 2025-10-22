import os
import time
import sys

import torch
import numpy as np
from scipy.sparse import coo_matrix

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .FPSRModel import FPSRModel
from elliot.utils import logging as logging_project

logger = logging_project.get_logger("__main__")

class FPSR(RecMixin, BaseRecommenderModel):
    r"""
    FPSR: Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3543507.3583240>`_

    Args:
        eigen_dim: Number of eigenvectors extracted
        l_w: Regularization coefficient
        rho: Hyperparameter introduced by ADMM
        w_1: l_1 regularization term
        w_2: l_2 regularization term
        eta: regularizes the similarity between items within the partition
        eps: tolerance
        tau: Size ratio

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.FPSR:
          meta:
            save_recs: True
          eigen_dim: 256
          l_w: 0.2
          rho: 500
          w_1: 0.8
          w_2: 0.1
          eta: 1.0
          eps: 5e-3
          tau: 0.5
          seed: 2026
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._params_list = [
            ("_eigen_dim", "eigen_dim", "eigen_dim", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.5, float, None),
            ("_rho", "rho", "rho", 5000, int, None),
            ("_w_1", "w_1", "w_1", 0.1, float, None),
            ("_w_2", "w_2", "w_2", 1.0, float, None),
            ("_eta", "eta", "eta", 1.0, float, None),
            ("_tau", "tau", "tau", 0.2, float, None),
            ("_eps", "eps", "eps", 5e-3, float, None),
            ("_save_heatmap", "save_heatmap", "save_heatmap", False, bool, None)
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()

        self._inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                                 shape=(self._num_users, self._num_items))

        self._model = FPSRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            eigen_dim=self._eigen_dim,
            l_w=self._l_w,
            tau=self._tau,
            eta=self._eta,
            eps=self._eps,
            w_1=self._w_1,
            w_2=self._w_2,
            rho=self._rho,
            inter=self._inter,
            dataset=self._config.dataset,
            save_heatmap=self._save_heatmap,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "FPSR" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        start = time.time()
        self._model.initialize()
        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")
        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        with torch.no_grad():
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
