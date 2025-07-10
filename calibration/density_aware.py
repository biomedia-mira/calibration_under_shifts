"""
This file contains classes related to Density-Aware Calibration. 

Code from https://github.com/futakw/DensityAwareCalibration/blob/master/density_aware_calib.py
"""

from pathlib import Path
import numpy as np
from scipy import optimize
from sklearn.neighbors import NearestNeighbors
import time
from calibration.post_hoc_calibrators import one_hot


def np_softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x


class KNNScorer:
    """ """

    def __init__(self, output_dir: Path, n_neighbors: bool = 50) -> None:
        """
        Args:
            output_dir: Path to model directory, where to save distances
            n_neighbors: Number of neighbours to use for KNN.
        """
        self.output_dir = output_dir
        self.n_neighbors = n_neighbors

    def fit(self, train_features) -> None:
        """ """
        n_feats = train_features.shape[0]
        if n_feats > 50000:
            selected_indices = np.random.choice(
                np.arange(n_feats), min(n_feats, 50000), replace=False
            )
            train_features = train_features[selected_indices]
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=6)
        self.nn.fit(train_features)

    def get_ood_scores(
        self, ood_features, name_eval_loader: str, feature_idx=0
    ) -> np.ndarray:
        """
        Args:
            ood_results: Dict, as returned by run_inference
            name_eval_loader: str, name of data loader used for caching results to disk.
        """

        distances_ood = self.nn.kneighbors(ood_features, n_neighbors=self.n_neighbors)[
            0
        ][:, -1]

        return distances_ood


class DAC(object):
    def __init__(
        self,
        output_dir,
        number_features,
        suffix_filenames,
        tol=1e-12,
        eps=1e-7,
        disp=False,
        use_bias=True,
    ):
        """
        T = (w_i * knn_score_i) + w0
        p = softmax(logits / T)
        """
        self.method = "L-BFGS-B"

        self.number_features = number_features
        print("Number features: ", self.number_features)

        self.tol = tol
        self.eps = eps
        self.disp = disp

        self.use_bias = use_bias
        self.bnds = [[0, 10000.0]] * self.number_features
        self.init = [1.0] * self.number_features
        if self.use_bias:
            self.bnds += [[-100.0, 100.0]]
            self.init += [1.0]
        self.output_dir = output_dir
        self.suffix = suffix_filenames

    def fit_knn_scorers(self, train_features):
        self.knn_scorers = []
        for i, train_feats_i in enumerate(train_features):
            knn_score = KNNScorer(self.output_dir)
            knn_score.fit(train_feats_i)
            self.knn_scorers.append(knn_score)

    def get_temperature(self, w, ood_score):
        if self.number_features == 1:
            if type(ood_score).__module__ == np.__name__:
                if len(ood_score.shape) == 1:
                    ood_score = [ood_score]
                else:
                    ood_score = [ood_score[i, :] for i in range(ood_score.shape[0])]

        assert len(ood_score) == self.number_features, (
            ood_score,
            len(ood_score),
            self.number_features,
        )

        if len(ood_score) != 0:
            sample_size = len(ood_score[0])
            t = np.zeros(sample_size)

            for i in range(self.number_features):
                t += w[i] * ood_score[i]
            if self.use_bias:
                t += w[-1]
        else:
            # temperature scaling
            t = np.zeros(1)
            t += w[-1]

        # return t
        # temperature should be a positive value
        return np.clip(t, 1e-20, None)

    def mse_lf(self, w, *args):
        ## find optimal temperature with MSE loss function
        logit, label, ood_score = args
        t = self.get_temperature(w, ood_score)
        logit = logit / t[:, None]
        p = np_softmax(logit)
        mse = np.mean((p - label) ** 2)
        return mse

    def ll_lf(self, w, *args):
        ## find optimal temperature with Cross-Entropy loss function
        logit, label, ood_score = args
        t = self.get_temperature(w, ood_score)
        logit = logit / t[:, None]
        p = np_softmax(logit)
        N = p.shape[0]
        ce = -np.sum(label * np.log(p + 1e-12)) / N
        return ce

    def optimize(self, logit, label, feats, reference_loader_name="val", loss="ce"):
        """
        logit (N, C): classifier's outputs before softmax
        label (N,): true labels
        """
        ood_scores = self.get_all_distances(feats, reference_loader_name + self.suffix)
        if label.ndim == 1 or label.shape[1] == 1:
            label = one_hot(label, logit.shape[1])
        assert label.shape[1] == logit.shape[1]

        if not isinstance(self.eps, list):
            self.eps = [self.eps]

        if loss == "ce":
            func = self.ll_lf
        elif loss == "mse":
            func = self.mse_lf
        else:
            raise NotImplementedError

        # func:ll_t, 1.0:initial guess, args: args of the func, ..., tol: tolerence of minimization
        st = time.time()
        params = optimize.minimize(
            func,
            self.init,
            args=(logit, label, ood_scores),
            method=self.method,
            bounds=self.bnds,
            tol=self.tol,
            options={"eps": self.eps, "disp": self.disp},
        )
        ed = time.time()

        w = params.x
        print("DAC Optimization done!: ({} sec)".format(ed - st))
        if self.use_bias:
            print(f"T = {w[:-1]} * ood_score_i + {w[-1]}")
        else:
            print(f"T = {w[:-1]} * ood_score_i")

        optim_value = params.fun
        self.w = w

        return self.get_optim_params()

    def get_all_distances(self, feats, ood_loader_name):
        ood_scores = []
        for i in range(len(self.knn_scorers)):
            ood_scores.append(
                self.knn_scorers[i].get_ood_scores(
                    feats[i], ood_loader_name + self.suffix, i
                )
            )
        return ood_scores

    def calibrate(self, logits, feats, ood_loader_name):
        return np_softmax(
            self.calibrate_before_softmax(logits, feats, ood_loader_name), 1
        )

    def calibrate_before_softmax(self, logits, feats, ood_loader_name):
        ood_scores = self.get_all_distances(feats, ood_loader_name + self.suffix)
        w = self.w
        t = self.get_temperature(w, ood_scores)
        return logits / t[:, None]

    def get_optim_params(self):
        return self.w
