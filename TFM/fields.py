import logging
from os import stat
from types import DynamicClassAttribute

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from TFM import Vectors
from TFM.utils import (
    calc_Green,
    convert_complex_to_vectors,
    fft_for_vectors,
    get_Wavefunction_in_FS,
    is_edge,
)


class DPF(Vectors):
    @property
    def _constrctor(self):
        return DPF

    @classmethod
    def load_DPF(cls, path: str) -> "DPF":
        file = open(path, "r")
        data = file.read()
        data = [s.split()[:5] for s in data.split("\n")]
        df = pd.DataFrame(
            data, columns=["x", "y", "vx", "vy", "m"], dtype=np.float64
        ).dropna()
        return DPF(df).confirm()

    @classmethod
    def generate_fields(
        cls, nCol: int, nRow: int, size: int, mode: str, info: dict
    ) -> list:
        return [
            DPF(x)
            for x in Vectors.generate_fields(nCol, nRow, size, mode=mode, info=info)
        ]

    def fftc(
        self,
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ) -> "TFF":
        """
        Estimate traction force field from displacement field by Fourier transform traction cytometry (FFTC)
        https://www.cell.com/biophysj/pdf/S0006-3495(08)70780-X.pdf

        This program refers to https://github.com/qztseng/imagej_plugins/blob/master/current/src/FTTC/FTTC_.java
        Args:
            disXY (DPF): the target displacement field
            pixel (float): pixel length in micron
            mu (float): the poission ratio of gel
            E (float): Young's modulus of gel
            L (float): lambda parameter for the regularization kernel (0th order regularization)

        Returns:
            TFF: Estimated Traction Force Field
        """

        # get coordinate
        gridX = self.loc[:, "x"]
        gridY = self.loc[:, "y"]

        # get shape info
        dim = self.get_Dimensions()
        nCol, nRow, dPixel = dim
        D = dPixel * pixel

        # FFT
        disXCF = fft_for_vectors(DPF(self.copy() * pixel), target="vx")
        disYCF = fft_for_vectors(DPF(self.copy() * pixel), target="vy")
        disXCF[0, 0] = 0
        disYCF[0, 0] = 0

        # wave function in fourier space
        Kx = get_Wavefunction_in_FS(nCol, D)
        Ky = get_Wavefunction_in_FS(nRow, D)

        # calculate convolution

        # H is the regularization
        # G is Green function
        H = np.identity(2, dtype=np.complex) * L * L
        G = np.zeros([2, 2], dtype=np.complex)
        TractionXF = np.zeros([nRow, nCol], dtype=np.complex)
        TractionYF = np.zeros([nRow, nCol], dtype=np.complex)
        for i in range(len(Ky)):
            for j in range(len(Kx)):
                flag = is_edge(j, i, nCol, nRow)
                G = calc_Green(Kx[j], Ky[i], flag, mu, E)
                Gt = G.T
                G1 = Gt @ G
                G1 += H
                Ginv = np.linalg.inv(G1)

                dd = np.array([disXCF[i, j], disYCF[i, j]])
                GtU = Gt @ dd
                TXY = Ginv @ GtU
                TractionXF[i, j] = TXY[0]
                TractionYF[i, j] = TXY[1]
        TractionXF[0, 0] = 0
        TractionYF[0, 0] = 0

        # invert fft
        TractionXF = np.fft.ifft2(TractionXF)
        TractionYF = np.fft.ifft2(TractionYF)

        TractionXR = TractionXF.real.flatten()
        TractionYR = TractionYF.real.flatten()
        magnitude = np.sqrt(TractionXR ** 2 + TractionYR ** 2)
        df = pd.DataFrame(
            {
                "x": gridX,
                "y": gridY,
                "vx": TractionXR,
                "vy": TractionYR,
                "m": magnitude,
            }
        )
        return TFF(df).confirm()

    def get_Dimensions(self) -> list:
        return super().get_Dimensions()

    def confirm(self):
        return DPF(super().confirm())

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        # return the target data in 2D (x is column and y is row)
        return super().rearrange_by_coordinate(target)

    def draw(
        self,
        figsize: tuple = (5, 5),
        scale: int = None,
        save_img: bool = False,
        name: str = None,
    ):
        super().draw(figsize=figsize, scale=scale, save_img=save_img, name=name)


class TFF(Vectors):
    @property
    def _constructor(self):
        return TFF

    @classmethod
    def load_TFF(cls, path: str) -> "TFF":
        file = open(path, "r")
        data = file.read()
        data = [s.split() for s in data.split("\n")]
        df = pd.DataFrame(
            data, columns=["x", "y", "vx", "vy", "m"], dtype=np.float64
        ).dropna()
        return TFF(df).confirm()

    @classmethod
    def generate_fields(
        cls, nCol: int, nRow: int, size: int, mode: str = "cGL", info: dict = None
    ) -> list:
        return [
            TFF(x)
            for x in Vectors.generate_fields(nCol, nRow, size, mode=mode, info=info)
        ]

    @classmethod
    def kalman_FFTC(
        cls,
        data: list,
        mode: int,
        use_em: bool = False,
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ) -> "TFF":
        """
        Estimate traction force fields from a list of dipslacement fields by usign kalman-smoother
        Args:
            data (list[DPF]): list of displacement field
            mode: level of transition equation
            mu (float): the poission ratio of gel
            E (float): Young's modulus of gel
            L (float): lambda parameter for the regularization kernel (0th order regularization)
        Returns:
            list[TFF]: Estimated traction force fields
        """
        if mode == 0:
            return cls.kalman_FFTC_d0(
                data=data, use_em=use_em, pixel=pixel, mu=mu, E=E, L=L
            )
        elif mode == 1:
            return cls.kalman_FFTC_d1(
                data=data, use_em=use_em, pixel=pixel, mu=mu, E=E, L=L
            )

    @classmethod
    def kalman_FFTC_d0(
        cls,
        data: list,
        use_em: bool = False,
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ) -> "TFF":

        logging.info("Kalman smoother for T_(t+1) ~ T_t")
        initial_dpf = data[0]
        data = data[1:]
        nCol, nRow, dPixel = initial_dpf.get_Dimensions()
        D = dPixel * pixel

        # convert raw data to process kalman-filter in fourier space
        logging.info("Get train data")
        train = cls._get_train_data(data, pixel, 0)

        # convert initial_dpf for kalman filter
        # index for rearanging the results
        initial_state_vectors, tff_index = cls._get_initial_state_vector(initial_dpf)

        # set obervation matrix
        # beta_(t+1) ~ beta_t
        # the vectors should be arranged by (xi_real, yi_real, xi_imag, yi_imag)
        H = cls._set_observation_matrix(nCol, nRow, mode=0, D=D, mu=mu, E=E, L=L)
        F = cls._set_transition_matrix(initial_state_vectors.shape[0], mode=0)

        kf = KalmanFilter(
            n_dim_obs=initial_state_vectors.shape[0],
            n_dim_state=initial_state_vectors.shape[0],
            initial_state_mean=initial_state_vectors.values,
            initial_state_covariance=np.identity(initial_state_vectors.shape[0]),
            transition_matrices=F,
            observation_matrices=H,
            observation_covariance=np.identity(initial_state_vectors.shape[0]),
            transition_covariance=np.identity(initial_state_vectors.shape[0]),
        )
        if use_em:
            em_vars = [
                "initial_state_covariance",
                "observation_covariance",
                "transition_covariance",
            ]
            logging.info(f"EM-algorithm for {em_vars}")
            emed_kf = kf.em(
                train,
                em_vars=em_vars,
            )
            kf = emed_kf
        logging.info("Start kalman-smoother")
        smoothed_state_means, smoothed_state_covs = kf.smooth(train)
        logging.info("Done")

        # reconstruct traction force filed from complex matrix
        result = []
        for i in range(len(smoothed_state_means)):
            res_XR = smoothed_state_means[i, ::4]
            res_YR = smoothed_state_means[i, 1::4]
            res_XI = smoothed_state_means[i, 2::4]
            res_YI = smoothed_state_means[i, 3::4]
            resXCF = res_XR + 1j * res_XI
            resYCF = res_YR + 1j * res_YI
            res_XCF = pd.DataFrame(resXCF, index=tff_index)
            res_YCF = pd.DataFrame(resYCF, index=tff_index)

            res_TractionXF = np.fft.ifft2(res_XCF.unstack().values)
            res_TractionYF = np.fft.ifft2(res_YCF.unstack().values)
            res_TractionXR = res_TractionXF.real.flatten()
            res_TractionYR = res_TractionYF.real.flatten()
            res_magnitude = np.sqrt(res_TractionXR ** 2 + res_TractionYR ** 2)
            res = TFF(
                {
                    "x": initial_dpf.loc[:, "x"],
                    "y": initial_dpf.loc[:, "y"],
                    "vx": res_TractionXR,
                    "vy": res_TractionYR,
                    "m": res_magnitude,
                }
            ).confirm()
            result.append(res)
        return result

    @classmethod
    def kalman_FFTC_d1(
        cls,
        data: list,
        use_em: bool = False,
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ) -> "TFF":

        logging.info("Kalman smoother for T_(t+1) ~ T_t")
        initial_dpf0 = data[0]
        initial_dpf1 = data[1]
        data = data[1:]
        nCol, nRow, dPixel = initial_dpf0.get_Dimensions()
        D = dPixel * pixel

        # convert raw data to process kalman-filter in fourier space
        logging.info("Get train data")
        train = cls._get_train_data(data, pixel, 1)

        # convert initial dpf for kalman filter
        # index for rearanging the results
        # because of using differential, state vectors hold two vectors in time t and t-1
        initial_state_vectors1, tff_index = cls._get_initial_state_vector(initial_dpf0)
        initial_state_vectors2, _ = cls._get_initial_state_vector(initial_dpf1)
        initial_state_vectors = pd.concat(
            [initial_state_vectors1, initial_state_vectors2]
        )

        # set obervation matrix
        # beta_(t+1) - beta_t ~ beta_t - beta_(t-1)
        # the vectors should be arranged by (xi_real, yi_real, xi_imag, yi_imag)
        H = cls._set_observation_matrix(nCol, nRow, mode=1, D=D, mu=mu, E=E, L=L)
        F = cls._set_transition_matrix(initial_state_vectors.shape[0], mode=1)
        print(initial_state_vectors.shape)
        print(H.shape)
        print(F.shape)
        kf = KalmanFilter(
            n_dim_obs=initial_state_vectors.shape[0],
            n_dim_state=initial_state_vectors.shape[0],
            initial_state_mean=initial_state_vectors.values,
            initial_state_covariance=np.identity(initial_state_vectors.shape[0]),
            transition_matrices=F,
            observation_matrices=H,
            observation_covariance=np.identity(initial_state_vectors.shape[0]),
            transition_covariance=np.identity(initial_state_vectors.shape[0]),
        )
        if use_em:
            em_vars = [
                "initial_state_covariance",
                "observation_covariance",
                "transition_covariance",
            ]
            logging.info(f"EM-algorithm for {em_vars}")
            emed_kf = kf.em(
                train,
                em_vars=em_vars,
            )
            kf = emed_kf
        logging.info("Start kalman-smoother")
        smoothed_state_means, smoothed_state_covs = kf.smooth(train)
        logging.info("Done")

        # reconstruct traction force filed from complex matrix
        result = []
        for i in range(len(smoothed_state_means)):
            res_XR = smoothed_state_means[i, ::4]
            res_YR = smoothed_state_means[i, 1::4]
            res_XI = smoothed_state_means[i, 2::4]
            res_YI = smoothed_state_means[i, 3::4]
            resXCF = res_XR + 1j * res_XI
            resYCF = res_YR + 1j * res_YI
            res_XCF = pd.DataFrame(resXCF[: len(tff_index)], index=tff_index)
            res_YCF = pd.DataFrame(resYCF[: len(tff_index)], index=tff_index)

            res_TractionXF = np.fft.ifft2(res_XCF.unstack().values)
            res_TractionYF = np.fft.ifft2(res_YCF.unstack().values)
            res_TractionXR = res_TractionXF.real.flatten()
            res_TractionYR = res_TractionYF.real.flatten()
            res_magnitude = np.sqrt(res_TractionXR ** 2 + res_TractionYR ** 2)
            res = TFF(
                {
                    "x": initial_dpf0.loc[:, "x"],
                    "y": initial_dpf0.loc[:, "y"],
                    "vx": res_TractionXR,
                    "vy": res_TractionYR,
                    "m": res_magnitude,
                }
            ).confirm()
            result.append(res)
        return result

    def inv_fftc(
        self,
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
        noise_flag: bool = True,
        noise_ratio: float = 0.1,
    ) -> "DPF":
        """
        Convert traction force field to displacement field by inverting FFTC

        Returns:
            DPF: calculated displacement field
        """
        gridX = self.loc[:, "x"]
        gridY = self.loc[:, "y"]

        # the shape of _fft_for_vectors is 2D, so change to 1D
        tffXCF = pd.DataFrame(fft_for_vectors(self, "vx")).stack()
        tffYCF = pd.DataFrame(fft_for_vectors(self, "vy")).stack()

        nCol, nRow, dPixel = self.get_Dimensions()
        D = dPixel * pixel

        # wave function in fourier space
        Kx = get_Wavefunction_in_FS(nCol, D)
        Ky = get_Wavefunction_in_FS(nRow, D)

        G = np.zeros([2, 2], dtype=np.complex)
        disXCF = np.zeros([nRow, nCol], dtype=np.complex)
        disYCF = np.zeros([nRow, nCol], dtype=np.complex)

        for i in range(len(Ky)):
            for j in range(len(Kx)):
                flag = is_edge(j, i, nCol, nRow)
                G = calc_Green(Kx[j], Ky[i], flag, mu, E)
                dd = np.array([tffXCF[i, j], tffYCF[i, j]])
                TXY = G @ dd
                disXCF[i, j] = TXY[0]
                disYCF[i, j] = TXY[1]

        disXCF[0, 0] = 0
        disYCF[0, 0] = 0

        # invert fft
        disXF = np.fft.ifft2(disXCF)
        disYF = np.fft.ifft2(disYCF)

        disXR = disXF.real.flatten() / pixel
        disYR = disYF.real.flatten() / pixel
        magnitude = np.sqrt(disXR ** 2 + disYR ** 2)
        df = DPF(
            {
                "x": gridX,
                "y": gridY,
                "vx": disXR,
                "vy": disYR,
                "m": magnitude,
            }
        )

        if noise_flag:
            nCol, nRow, _ = df.get_Dimensions()
            mx = df.loc[:, "vx"].abs().mean() * noise_ratio
            my = df.loc[:, "vy"].abs().mean() * noise_ratio
            df.loc[:, "vx"] += np.random.normal(0, mx, nCol * nRow)
            df.loc[:, "vy"] += np.random.normal(0, my, nCol * nRow)
        return df.confirm()

    @staticmethod
    def _set_observation_matrix(
        nCol: int,
        nRow: int,
        mode: int,
        D: float,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ):
        Kx = get_Wavefunction_in_FS(nCol, D)
        Ky = get_Wavefunction_in_FS(nRow, D)
        H = np.zeros([4 * nCol * nRow, 4 * nCol * nRow], dtype=np.float64)
        for i in range(nRow):
            for j in range(nCol):
                o = (i * nCol + j) * 4
                flag = is_edge(j, i, nCol, nRow)
                G = calc_Green(Kx[j], Ky[i], flag, mu, E)
                H[o : o + 2, o : o + 2] = G
                H[o + 2 : o + 4, o + 2 : o + 4] = G
        H[0:4, 0:4] = np.zeros([4, 4], dtype=np.float64)
        if mode == 1:
            H = np.block([[H, np.zeros(H.shape)], [np.zeros(H.shape), H]])
        return H

    @staticmethod
    def _set_transition_matrix(L: int, mode: int):
        if mode == 0:
            return np.identity(L)
        elif mode == 1:
            l = L // 2
            return np.block(
                [
                    [np.zeros((l, l)), np.identity(l)],
                    [np.identity(l) * -1, np.identity(l) * 2],
                ]
            )

    @staticmethod
    def _get_train_data(ls: list, pixel: float, mode: int):
        res = []
        for pdf in ls:
            df = TFF(pdf.copy())
            df.loc[:, "vx"] *= pixel
            df.loc[:, "vy"] *= pixel
            disXCF = pd.DataFrame(fft_for_vectors(df, "vx")).stack()
            disXR, disXI = convert_complex_to_vectors(disXCF)
            disYCF = pd.DataFrame(fft_for_vectors(df, "vy")).stack()
            disYR, disYI = convert_complex_to_vectors(disYCF)
            obsCF = (
                pd.concat([disXR, disYR, disXI, disYI]).sort_index().astype("float64")
            )
            res.append(obsCF.values)
        if mode == 1:
            res_ = []
            for i in range(0, len(res) - 1):
                v = np.concatenate([res[i], res[i + 1]])
                res_.append(v)
            res = res_
        return res

    @staticmethod
    def _get_initial_state_vector(initial_dpf: "DPF"):
        initial_tff = initial_dpf.fftc()
        tffXCF = pd.DataFrame(fft_for_vectors(initial_tff, "vx")).stack()
        tffYCF = pd.DataFrame(fft_for_vectors(initial_tff, "vy")).stack()

        # split complex data to real part and imaginary part
        tffXR, tffXI = convert_complex_to_vectors(tffXCF)
        tffYR, tffYI = convert_complex_to_vectors(tffYCF)

        # sort by xi_real, yi_real, xi_imag, yi_imag
        initial_state_vectors = (
            pd.concat([tffXR, tffYR, tffXI, tffYI]).sort_index().astype("float64")
        )
        return initial_state_vectors, tffXCF.index

    def get_Dimensions(self) -> list:
        return super().get_Dimensions()

    def confirm(self):
        return TFF(super().confirm())

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        # return the target data in 2D (x is column and y is row)
        return super().rearrange_by_coordinate(target)

    def draw(
        self,
        figsize: tuple = (5, 5),
        scale: int = None,
        save_img: bool = False,
        name: str = None,
    ):
        super().draw(figsize=figsize, scale=scale, save_img=save_img, name=name)
