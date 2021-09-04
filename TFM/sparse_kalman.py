import dataclasses
import pickle
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, linalg
from typing import Tuple, List, Union


@dataclasses.dataclass
class SparseKalman:
    observation_matrix: Union[csr_matrix, List[csr_matrix]]
    observation_noise: Union[csr_matrix, List[csr_matrix]]
    transition_matrix: Union[csr_matrix, List[csr_matrix]]
    transition_noise: Union[csr_matrix, List[csr_matrix]]
    predicted_means: List[csr_matrix] = None
    predicted_covariances: List[csr_matrix] = None
    smoothed_means: List[csr_matrix] = None
    smoothed_covariances: List[csr_matrix] = None

    @staticmethod
    def _filter(
        G: csr_matrix,
        W: csr_matrix,
        F: csr_matrix,
        V: csr_matrix,
        m: csr_matrix,
        C: csr_matrix,
        y: csr_matrix,
    ) -> Tuple[csr_matrix, csr_matrix]:
        """Kalman Filter

        Args:
            G (csr_matrix): transition matrix
            W (csr_matrix): transition noise
            F (csr_matrix): observation matrix
            V (csr_matrix): observation matrix
            m (csr_matrix): predicted mean
            C (csr_matrix): predicted covariance
            y (csr_matrix): observed values

        Returns:
            [tuple]: (predicted mean, predicted covariance)
        """
        a = G * m
        R = G * C * G.T + W
        f = F * a
        Q = F * R * F.T + V
        # K = (np.linalg.solve(Q.T, F * R.T)).T
        K = R * F.T * linalg.inv(Q)
        K = csr_matrix(K)
        m = a + K * (y - f)
        C = R - K * F * R
        return m, C

    @staticmethod
    def _prediction(a, R, G, W):
        """Kalman prediction
        """
        a = G * a
        R = G * R * G.T + W
        return a, R

    @staticmethod
    def _smoothing(
        s: csr_matrix,
        S: csr_matrix,
        m: csr_matrix,
        C: csr_matrix,
        G: csr_matrix,
        W: csr_matrix,
    ) -> Tuple[csr_matrix, csr_matrix]:
        """Kalman smoothing

        Args:
            s (csr_matrix): smoothed mean
            S (csr_matrix): smoothed covariance
            m (csr_matrix): predicted mean
            C (csr_matrix): predicted covariance
            G (csr_matrix): transition matrix
            W (csr_matrix): transition noise

        Returns:
            Tuple[csr_matrix, csr_matrix]: (smoothed mean, smoothed covariance)
        """
        # 1時点先予測分布のパラメータ計算
        a = G * m
        R = G * C * G.T + W

        A = C * G.T * linalg.inv(R)
        A = csr_matrix(A)

        s = m + A * (s - a)
        S = C + A * (S - R) * A.T
        return s, S

    def Filter(
        self,
        data: List[csr_matrix],
        initial_mean: csr_matrix,
        initial_covariance: csr_matrix,
    ):
        T = len(data)
        predicted_means = []
        predicted_covariances = []
        for t in tqdm(range(T)):
            if t == 0:
                m, C = self._filter(
                    G=self.transition_matrix,
                    W=self.transition_noise,
                    F=self.observation_matrix,
                    V=self.observation_noise,
                    m=initial_mean,
                    C=initial_covariance,
                    y=data[t],
                )
                predicted_means.append(m)
                predicted_covariances.append(C)
            else:
                m, C = self._filter(
                    G=self.transition_matrix,
                    W=self.transition_noise,
                    F=self.observation_matrix,
                    V=self.observation_noise,
                    m=m,
                    C=C,
                    y=data[t],
                )
                predicted_means.append(m)
                predicted_covariances.append(C)
        self.predicted_means = predicted_means
        self.predicted_covariances = predicted_covariances

    def Smoother(self):
        if self.predicted_means is None or self.predicted_covariances is None:
            raise ValueError("Kalman Filtering has not done")
        T = len(self.predicted_means)
        smoothed_means = [0] * T
        smoothed_covariances = [0] * T
        for t in tqdm(range(T)):
            t = T - t - 1
            if t == T - 1:
                smoothed_means[t] = self.predicted_means[t]
                smoothed_covariances[t] = self.predicted_covariances[t]
            else:
                smoothed_means[t], smoothed_covariances[t] = self._smoothing(
                    s=smoothed_means[t + 1],
                    S=smoothed_covariances[t + 1],
                    m=self.predicted_means[t],
                    C=self.predicted_covariances[t],
                    G=self.transition_matrix,
                    W=self.transition_noise,
                )
        self.smoothed_means = smoothed_means
        self.smoothed_covariances = smoothed_covariances

    def export_predictions(self, is_np: bool = True):
        if is_np:
            return (
                np.array(list(map(lambda x: x.toarray(), self.predicted_means))),
                np.array(list(map(lambda x: x.toarray(), self.predicted_covariances))),
            )
        else:
            return self.predicted_means, self.predicted_covariances

    def export_smoothings(self, is_np: bool = True):
        if is_np:
            return (
                np.array(list(map(lambda x: x.toarray(), self.smoothed_means))),
                np.array(list(map(lambda x: x.toarray(), self.smoothed_covariances))),
            )
        else:
            return self.smoothed_means, self.smoothed_covariances

    def to_pickle(self, name: str = None):
        if name is None:
            name = "result"
        with open(f"{name}.skrs", "wb") as f:
            pickle.dump(self.copy(), f)

    @classmethod
    def from_pickle(self, name: str) -> "SparseKalman":
        with open(f"{name}.skrs", "rb") as f:
            res = pickle.load(f)
        return res
