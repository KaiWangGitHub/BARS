from scipy.stats import norm, binom_test
import numpy as np
from math import ceil, floor
from statsmodels.stats.proportion import proportion_confint
from typing import *

import torch

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [feature_size]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1))
                noise = torch.randn_like(batch, device=batch.device) * self.sigma
                predictions = self.base_classifier((batch + noise).unsqueeze(1).unsqueeze(3)).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]

# Certifying multiple samples in parallel
class Smooth2(Smooth):
    def __init__(self, base_classifier: torch.nn.Module, d: int, num_classes: int, noise_generator: object, device: object):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes: the number of classes, Kitsune: 2, CADE: 2, ACID: 4
        :param noise_generator: optimized noise generator
        :param d: the number of feature vector dimensions
        :param device: cpu / cuda
        """
        self.base_classifier = base_classifier
        self.d = d
        self.num_classes = num_classes
        self.noise_generator = noise_generator
        self.device = device

        assert noise_generator.d == self.d
        assert noise_generator.device == self.device

        self.eps = 1e-16

    def _certify2(self, x: torch.tensor, n0: int, n: int, t: float, alpha: float, batch_size_memory: int) -> (np.ndarray, np.ndarray):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [batch_size_iteration x feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size_memory: maximum batch size in memory for parallel smoothing
        :return: (predicted class: np.ndarray, certified normalized robustness radius: np.ndarray)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise2(x, n0, t, batch_size_memory)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise2(x, n, t, batch_size_memory)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[np.arange(cAHat.shape[0]), cAHat]
        pABar = self._lower_confidence_bound2(nA, n, alpha)
        # use pA lower bound to calculate normalized robustness radius
        radius_norm = norm.ppf(pABar)

        # when pABar < 0.5, abstain from making robustness certification
        idx_abstain = np.where(pABar < 0.5)[0]
        cAHat[idx_abstain] = self.ABSTAIN
        radius_norm[idx_abstain] = 0

        return cAHat, radius_norm

    def _sample_noise2(self, x: torch.tensor, num: int, t: float, batch_size_memory: int) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [batch_size_iteration x feature_size]
        :param num: number of samples to collect
        :param t: scale factor
        :param batch_size_memory: maximum batch size in memory for parallel smoothing
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        # batch size for iteration should be less than or equal to maximum batch size in memory
        assert x.size(0) <= batch_size_memory
        with torch.no_grad():
            counts = np.zeros((x.size(0), self.num_classes), dtype=int)
            while num > 0:
                batch_size_per_example = min(floor(batch_size_memory / x.size(0)), num)
                num -= batch_size_per_example

                batch = x.repeat((batch_size_per_example, 1))
                noise = self.noise_generator.sample_feat(x.size(0) * batch_size_per_example) * t
                predictions = self.base_classifier(batch, noise)
                counts += self._count_arr2(predictions.cpu().numpy(), x.size(0))

            return counts

    def _count_arr2(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros((length, self.num_classes), dtype=int)
        arr = arr.reshape(-1, length).T
        for c in range(self.num_classes):
            counts[:, c] += np.array(arr == c, dtype=int).sum(1)
        return counts

    def _lower_confidence_bound2(self, NA: np.ndarray, N: int, alpha: float) -> np.ndarray:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes" for each example
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: an ndarray of lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return np.array([proportion_confint(NA_, N, alpha=2 * alpha, method="beta")[0] for NA_ in NA])

    def bars_certify(self, x: torch.tensor, n0: int, n: int, t: float, alpha: float, batch_size_memory: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """ 
        :param x: the input [batch_size_iteration x feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size_memory: maximum batch size in memory for parallel smoothing
        :return: (predicted class: np.ndarray, 
                 certified dimension-wise robustness radius vector (feature space): np.ndarray,
                 certified dimension-heterogeneous robustness radius (feature space): np.ndarray)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        cAHat, radius_norm = self._certify2(x, n0, n, t, alpha, batch_size_memory)
        radius_norm_dim = torch.tensor(radius_norm).unsqueeze(1).repeat((1, self.d)).to(self.device)
        radius_feat_dim = self.noise_generator.norm_to_feat(radius_norm_dim).cpu().numpy() * t
        radius_feat = radius_feat_dim.mean(1)

        return cAHat, radius_feat_dim, radius_feat

class Noise(object):
    def __init__(self, distribution_transformer: torch.nn.Module, d: int, device: object):
        """
        distribution_transformer: Distribution Transformer model
        d: the number of feature vector dimensions
        device: cpu / cuda
        """
        self.distribution_transformer = distribution_transformer
        self.d = d
        self.device = device

    def sample_norm(self, n: int) -> torch.tensor:
        return torch.randn((n, self.d), device=self.device)

    def norm_to_feat(self, z: torch.tensor) -> torch.tensor:
        return self.distribution_transformer(z).to(self.device)

    def sample_feat(self, n: int) -> torch.tensor:
        z = self.sample_norm(n)
        return self.norm_to_feat(z)

    def get_weight(self) -> torch.tensor:
        return self.distribution_transformer.get_weight()