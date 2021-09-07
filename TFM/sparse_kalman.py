import dataclasses
import logging
from os import stat
import pickle
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr, csr_matrix, linalg
from typing import ClassVar, Tuple, List, Union


@dataclasses.dataclass
class SparseKalman:
    observation_matrix: Union[csr_matrix, List[csr_matrix]]
    observation_noise: Union[csr_matrix, List[csr_matrix]]
    transition_matrix: Union[csr_matrix, List[csr_matrix]]
    transition_noise: Union[csr_matrix, List[csr_matrix]]
    initial_mean: csr_matrix
    initial_covariance: csr_matrix
    predicted_means: List[csr_matrix] = None
    predicted_covariances: List[csr_matrix] = None
    kalman_gains: List[csr_matrix] = None
    smoothed_means: List[csr_matrix] = None
    smoothed_covariances: List[csr_matrix] = None
    kalman_smoothing_gains: List[csr_matrix] = None

    @staticmethod
    def _filter(
        G: csr_matrix,
        W: csr_matrix,
        F: csr_matrix,
        V: csr_matrix,
        m: csr_matrix,
        C: csr_matrix,
        y: csr_matrix,
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
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
            [tuple]: (predicted mean, predicted covariance, kalman gain)
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
        return m, C, K

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
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        """Kalman smoothing

        Args:
            s (csr_matrix): smoothed mean
            S (csr_matrix): smoothed covariance
            m (csr_matrix): predicted mean
            C (csr_matrix): predicted covariance
            G (csr_matrix): transition matrix
            W (csr_matrix): transition noise

        Returns:
            Tuple[csr_matrix, csr_matrix, csr_matrix]: (smoothed mean, smoothed covariance, kalman smoothing gain)
        """
        # 1時点先予測分布のパラメータ計算
        a = G * m
        R = G * C * G.T + W

        A = C * G.T * linalg.inv(R)
        A = csr_matrix(A)

        s = m + A * (s - a)
        S = C + A * (S - R) * A.T
        return s, S, A

    def Filter(
        self, data: List[csr_matrix],
    ):
        T = len(data)
        predicted_means = []
        predicted_covariances = []
        kalman_gains = []
        for t in tqdm(range(T)):
            if t == 0:
                m, C, k = self._filter(
                    G=self.transition_matrix,
                    W=self.transition_noise,
                    F=self.observation_matrix,
                    V=self.observation_noise,
                    m=self.initial_mean,
                    C=self.initial_covariance,
                    y=data[t],
                )
                predicted_means.append(m)
                predicted_covariances.append(C)
                kalman_gains.append(k)
            else:
                m, C, k = self._filter(
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
                kalman_gains.append(k)
        self.predicted_means = predicted_means
        self.predicted_covariances = predicted_covariances
        self.kalman_gains = kalman_gains
        return self

    def Smoother(self):
        if self.predicted_means is None or self.predicted_covariances is None:
            raise ValueError("Kalman Filtering has not done")
        T = len(self.predicted_means)
        smoothed_means = [0] * T
        smoothed_covariances = [0] * T
        kalman_smoothing_gains = [0] * (T - 1)
        for t in tqdm(reversed(range(T))):
            if t == T - 1:
                smoothed_means[t] = self.predicted_means[t]
                smoothed_covariances[t] = self.predicted_covariances[t]
            else:
                (
                    smoothed_means[t],
                    smoothed_covariances[t],
                    kalman_smoothing_gains[t],
                ) = self._smoothing(
                    s=smoothed_means[t + 1],
                    S=smoothed_covariances[t + 1],
                    m=self.predicted_means[t],
                    C=self.predicted_covariances[t],
                    G=self.transition_matrix,
                    W=self.transition_noise,
                )
        self.smoothed_means = smoothed_means
        self.smoothed_covariances = smoothed_covariances
        self.kalman_smoothing_gains = kalman_smoothing_gains
        return self

    def em(
        self, data, option: List[str] = None, n_iter: int = 10,
    ):
        logging.info("Start EM Algorithm ... ")
        mdl = self
        option = set(option)
        for _ in range(n_iter):
            mdl = mdl.Filter(data).Smoother()
            pairwise_covariances = mdl._smoothing_pairwise(
                mdl.smoothed_covariances, mdl.kalman_smoothing_gains
            )
            params = mdl._em(
                data=data,
                transition_matrix=mdl.transition_matrix,
                smoothed_means=mdl.smoothed_means,
                smoothed_covariances=mdl.smoothed_covariances,
                pairwise_covariances=pairwise_covariances,
            )
            if "transition matrix" in option:
                mdl.transition_matrix = params["transition matrix"]
            if "transition noise" in option:
                mdl.transition_noise = params["transition noise"]
            if "observation noise" in option:
                mdl.observation_noise = params["observation noise"]
            if "initial mean" in option:
                mdl.initial_mean = params["initial mean"]
            if "initial covariance" in option:
                mdl.initial_covariance = params["initial covariance"]
        logging.info("DONE !")
        return mdl

    @staticmethod
    def _outer(x: csr_matrix, y: csr_matrix) -> csr_matrix:
        x

    @staticmethod
    def _smoothing_pairwise(
        smoothed_covariances: List[csr_matrix], kalman_smoothing_gains: List[csr_matrix]
    ) -> List[csr_matrix]:
        T = len(smoothed_covariances)
        pairwise_covariances = [0] * T
        for t in range(1, T):
            pairwise_covariances[t] = (
                smoothed_covariances[t] * kalman_smoothing_gains[t - 1].T
            )
        return pairwise_covariances

    @staticmethod
    def _em_observation_noise(
        data: List[csr_matrix],
        G: csr_matrix,
        smoothed_means: List[csr_matrix],
        smoothed_covariances: List[csr_matrix],
    ) -> csr_matrix:
        """EM algorithm for observation noise

        Args:
            data (List[csr_matrix]): observations
            G (csr_matrix): transition matrix
            smoothed_means (List[csr_matrix]): smoothed means
            smoothed_covariances (List[csr_matrix]): smoothed covariances
        Returns:
            EM observation noise (csr_matrix)
        """
        T = len(data)
        d_obs = data[0].shape[0]
        res = csr_matrix(np.zeros((d_obs, d_obs)))
        for t in range(T):
            err = data[t] - G * smoothed_means[t]
            res += err * err.T + G * smoothed_covariances[t] * G.T
        return 1.0 / T * res

    @staticmethod
    def _em_transition_matrix(
        smoothed_means: List[csr_matrix],
        smoothed_covariances: List[csr_matrix],
        pairwise_covariances: List[csr_matrix],
    ):
        T = len(smoothed_means)
        d_state, _ = smoothed_means[0].shape
        res1 = csr_matrix(np.zeros((d_state, d_state)))
        res2 = csr_matrix(np.zeros((d_state, d_state)))

        for t in range(1, T):
            res1 += (
                pairwise_covariances[t] + smoothed_means[t] * smoothed_means[t - 1].T
            )

            res2 += (
                smoothed_covariances[t - 1]
                + smoothed_means[t - 1] * smoothed_means[t - 1].T
            )
        return res1 * linalg.inv(res2)

    @staticmethod
    def _em_transition_noise(
        G: csr_matrix,
        smoothed_means: List[csr_matrix],
        smoothed_covariances: List[csr_matrix],
        pairwise_covariances: List[csr_matrix],
    ) -> csr_matrix:
        """EM algorithm for transtion noise

        Args:
            G (csr_matrix): transition matrix
            smoothed_means (List[csr_matrix]): smoothed means
            smoothed_covariances (List[csr_matrix]): smoothed covariances
            pairwise_covariances (List[csr_matrix]): pairwise_covariances

        Returns:
            csr_matrix: EM transition noise
        """
        T = len(smoothed_means)
        d_state = smoothed_means[0].shape[0]
        res = csr_matrix(np.zeros((d_state, d_state)))
        for t in range(T - 1):
            err = smoothed_means[t + 1] - G * smoothed_means[t]
            Vt1t_A = pairwise_covariances[t + 1] * G.T
            res += (
                err * err.T
                + G * smoothed_covariances[t] * G.T
                + smoothed_covariances[t + 1]
                - Vt1t_A
                - Vt1t_A.T
            )

        return 1.0 / (T - 1) * res

    @staticmethod
    def _em_initial_state_mean(smoothed_means: List[csr_matrix]) -> csr_matrix:
        return smoothed_means[0]

    @staticmethod
    def _em_initial_state_covariance(
        initial_mean: csr_matrix,
        smoothed_means: List[csr_matrix],
        smoothed_covariances: List[csr_matrix],
    ) -> csr_matrix:
        x0 = smoothed_means[0]
        x0_x0 = smoothed_covariances[0] - x0 * x0.T
        return (
            x0_x0
            - initial_mean * x0.T
            - x0 * initial_mean.T
            + initial_mean * initial_mean.T
        )

    @staticmethod
    def _em(
        data: List[csr_matrix],
        transition_matrix: csr_matrix,
        smoothed_means: List[csr_matrix],
        smoothed_covariances: List[csr_matrix],
        pairwise_covariances: List[csr_matrix],
    ) -> dict:
        results = {}
        observatino_noise = SparseKalman._em_observation_noise(
            data,
            G=transition_matrix,
            smoothed_means=smoothed_means,
            smoothed_covariances=smoothed_covariances,
        )
        # transition_matrix = SparseKalman._em_transition_matrix(
        #     smoothed_means=smoothed_means,
        #     smoothed_covariances=smoothed_covariances,
        #     pairwise_covariances=pairwise_covariances,
        # )
        transition_noise = SparseKalman._em_transition_noise(
            G=transition_matrix,
            smoothed_means=smoothed_means,
            smoothed_covariances=smoothed_covariances,
            pairwise_covariances=pairwise_covariances,
        )
        initial_mean = SparseKalman._em_initial_state_mean(
            smoothed_means=smoothed_means
        )
        initial_covariance = SparseKalman._em_initial_state_covariance(
            initial_mean=initial_mean,
            smoothed_means=smoothed_means,
            smoothed_covariances=smoothed_covariances,
        )

        results["observation noise"] = observatino_noise
        results["transition matrix"] = transition_matrix
        results["transition noise"] = transition_noise
        results["initial mean"] = initial_mean
        results["initial covariance"] = initial_covariance
        return results

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
            pickle.dump(self, f)

    @classmethod
    def from_pickle(self, name: str) -> "SparseKalman":
        with open(f"{name}.skrs", "rb") as f:
            res = pickle.load(f)
        return res
