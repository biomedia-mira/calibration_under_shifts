"""
This file contains all the post-hoc calibrators implementation.

Code is adapted from https://github.com/mijoo308/Energy-Calibration/blob/main/source/calibration.py
and https://github.com/zhang64-llnl/Mix-n-Match-Calibration
"""

import numpy as np
from scipy import optimize
from scipy.special import softmax, logsumexp
from sklearn.isotonic import IsotonicRegression
import torch
from torch.nn import functional as F

from scipy.stats import norm


def one_hot(label, num_class):
    y = torch.zeros((label.shape[0], num_class)).long()
    label = torch.Tensor(label).to(torch.int64)
    y[label != -1] = F.one_hot(label[label != -1], num_class)
    return y.numpy()


class TemperatureScaler:
    def fit(self, logit, label):
        bnds = ((0.05, 20.0),)
        print(f"Before {self._ll_t(1.0, logit, label)}")
        t = optimize.minimize(
            self._ll_t,
            1.0,
            args=(logit, label),
            method="L-BFGS-B",
            bounds=bnds,
            tol=1e-12,
            options={"disp": False},
        )
        print(f"After {self._ll_t(t.x, logit, label)}")
        self.t = t.x
        print(f"Learned TS: {self.t}")

    def calibrate(self, logit):
        logit = logit / self.t
        p = softmax(logit, axis=1)
        return p

    def _ll_t(self, t, *args):
        logit, label = args
        label = one_hot(label, logit.shape[1]).astype(float)
        label[label.sum(1) == 0] = (
            np.ones_like(label[label.sum(1) == 0]) / logit.shape[1]
        )
        logit = logit / t
        p = np.clip(softmax(logit, axis=1), 1e-20, 1 - 1e-20)
        N = p.shape[0]
        ce = -np.sum(label * np.log(p)) / N
        return ce


class ETSCalibrator:
    def __init__(self, t, n_class):
        self.t = t
        self.n_class = n_class

    def _ll_w(self, w, *args):
        p0, p1, p2, label = args
        label = one_hot(label, p0.shape[1])
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        N = p.shape[0]
        ce = -np.sum(label * np.log(p)) / N
        return ce

    def fit(self, logit, label):
        p1 = softmax(logit, axis=1)
        logit = logit / self.t
        p0 = softmax(logit, axis=1)
        p2 = np.ones_like(p0) / self.n_class

        bnds_w = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        )

        def my_constraint_fun(x):
            return np.sum(x) - 1

        constraints = {
            "type": "eq",
            "fun": my_constraint_fun,
        }

        w = optimize.minimize(
            self._ll_w,
            (1.0, 0.0, 0.0),
            args=(p0, p1, p2, label),
            method="SLSQP",
            constraints=constraints,
            bounds=bnds_w,
            tol=1e-12,
            options={"disp": False},
        )

        self.w = w.x

    def calibrate(self, logit):
        p1 = softmax(logit, axis=1)
        logit = logit / self.t
        p0 = softmax(logit, axis=1)
        p2 = np.ones_like(p0) / self.n_class
        p = self.w[0] * p0 + self.w[1] * p1 + self.w[2] * p2
        return p


class IRMCalibrator:
    def fit(self, logits, labels):
        labels = one_hot(labels, logits.shape[1])
        p = softmax(logits, axis=1)
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.ir.fit(p.flatten(), (labels.flatten()))

    def calibrate(self, logits):
        p_eval = softmax(logits, axis=1)
        yt_ = self.ir.predict(p_eval.flatten())
        p = yt_.reshape(logits.shape) + 1e-9 * p_eval
        return p


class IROvACalibrator:
    def fit(self, logits, labels):
        labels = one_hot(labels, logits.shape[1])
        p = softmax(logits, axis=1)
        self.list_ir = []
        for ii in range(p.shape[1]):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p[:, ii].astype("double"), labels[:, ii].astype("double"))
            self.list_ir.append(ir)

    def calibrate(self, logits):
        p_eval = softmax(logits, axis=1)
        for ii in range(p_eval.shape[1]):
            ir = self.list_ir[ii]
            p_eval[:, ii] = ir.predict(p_eval[:, ii])
        return p_eval


class IROvATSCalibrator:
    def __init__(self, t):
        self.t = t
        self.irova_calibrator = IROvACalibrator()

    def fit(self, logits, labels):
        logits = logits / self.t
        self.irova_calibrator.fit(logits, labels)

    def calibrate(self, logits):
        logits = logits / self.t
        return self.irova_calibrator.calibrate(logits)


class EBSCalibrator:
    """
    [ Energy Based Instance-wise Calibration ]
    """

    def __init__(self, initial_t):
        self.t = initial_t

    def fit(self, logits, labels, ood_logits):
        energy = -(logsumexp(logits, axis=1))

        # (1) correct energy pdf
        o_indices = np.argmax(softmax(logits, axis=1), axis=1) == labels
        o_samples = energy[o_indices]
        o_mu, o_sigma = norm.fit(o_samples)
        self.o_pdf = norm(o_mu, o_sigma)

        # (2) incorrect energy pdf
        labels = one_hot(labels, logits.shape[1])
        x_samples = energy[~(o_indices)]
        if ood_logits is not None:
            ood_energy = -logsumexp(ood_logits, axis=1)
            x_samples = np.concatenate((x_samples, ood_energy))
            logits = np.concatenate((logits, ood_logits))
            labels = np.concatenate(
                (labels, np.zeros((ood_logits.shape[0], logits.shape[1])))
            )

        x_mu, x_sigma = norm.fit(x_samples)
        self.x_pdf = norm(x_mu, x_sigma)

        shuffled_indices = np.random.permutation(logits.shape[0])
        logits = logits[shuffled_indices]
        labels = labels[shuffled_indices]

        bnds_theta = ((0.0, 250.0), (0.0, 250.0))
        print(f"Before theta: {self._mse_ebs((0.0, 0.0), logits, labels)}")
        theta = optimize.minimize(
            self._mse_ebs,
            (0.0, 0.0),
            args=(logits, labels),
            method="L-BFGS-B",
            bounds=bnds_theta,
            tol=1e-12,
            options={"disp": False},
        )

        theta = theta.x
        print("EBS theta : ", theta)
        print(f"After theta: {self._mse_ebs(theta, logits, labels)}")
        self.theta = theta

    def calibrate(self, logits):
        energy = -logsumexp(logits, axis=1)
        o_likelihood = self.o_pdf.pdf(energy)
        x_likelihood = self.x_pdf.pdf(energy)
        logits = (
            logits
            / (self.t - o_likelihood * self.theta[0] + x_likelihood * self.theta[1])[
                :, np.newaxis
            ]
        )
        p_eval = softmax(logits, axis=1)
        return p_eval

    def _mse_ebs(self, theta, *args):
        logit, label = args
        energy = -(logsumexp(logit, axis=1))

        o_likelihood = self.o_pdf.pdf(energy)
        x_likelihood = self.x_pdf.pdf(energy)
        logit = (
            logit
            / (self.t - o_likelihood * theta[0] + x_likelihood * theta[1])[
                :, np.newaxis
            ]
        )
        p = softmax(logit, axis=1)
        mse = np.mean((p - label) ** 2)
        return mse
