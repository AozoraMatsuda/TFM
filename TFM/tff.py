from os import stat
from types import DynamicClassAttribute

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from TFM import DPF, Vectors


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
    def FFTC(
        cls,
        disXY: "DPF",
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ) -> "TFF":

        dim = disXY.get_Dimensions()

        # get disformation and coordinate
        disX = disXY.rearrange_by_coordinate("vx")
        disX *= pixel
        disY = disXY.rearrange_by_coordinate("vy")
        disY *= pixel
        gridX = disXY.loc[:, "x"]
        gridY = disXY.loc[:, "y"]

        # real distance between points
        dPixel = dim[2]
        D = dPixel * pixel

        # get shape info
        nCol = disX.shape[1]
        nRow = disX.shape[0]

        # FFT
        disXCF = np.fft.fft2(disX)
        disYCF = np.fft.fft2(disY)
        disXCF[0, 0] = 0
        disYCF[0, 0] = 0

        # wave function in fourier space
        Kx = cls._get_Wavefunction_in_FS(nCol, D)
        Ky = cls._get_Wavefunction_in_FS(nRow, D)

        # calculate convolution
        H = np.identity(2, dtype=np.complex) * L * L
        G = np.zeros([2, 2], dtype=np.complex)
        TractionXF = np.zeros([nRow, nCol], dtype=np.complex)
        TractionYF = np.zeros([nRow, nCol], dtype=np.complex)
        for i in range(len(Ky)):
            for j in range(len(Kx)):
                flag = cls._is_edge(j, i, nCol, nRow)
                G = cls._calc_Green(Kx[j], Ky[i], flag, mu, E)
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

    @staticmethod
    def _calc_Green(
        kx: float, ky: float, is_edge: bool, mu: float = 0.5, E: float = 5000,
    ) -> np.array:
        G = np.zeros([2, 2], dtype=np.complex)
        k = np.sqrt(kx * kx + ky * ky)
        if is_edge:
            G[0, 1] = 0
            G[1, 0] = 0
        else:
            gg = -mu * kx * ky
            G[0, 1] = gg
            G[1, 0] = gg
        G0 = 2 * (1 + mu) / (E * pow(k, 3))
        G[0, 0] = (1 - mu) * (k * k) + mu * (ky * ky)
        G[1, 1] = (1 - mu) * (k * k) + mu * (kx * kx)
        G *= G0
        return G

    @staticmethod
    def _get_Wavefunction_in_FS(num: int, D: int) -> np.array:
        # return wave function in fourier space
        ls = (
            (2 * np.pi)
            / (D * num)
            * np.hstack(
                [
                    np.arange(0, num // 2 + 1, 1),
                    (-1) * np.arange(np.round(num / 2) - 1, 0, -1),
                ]
            )
        )
        return ls

    @staticmethod
    def _is_edge(x: int, y: int, nCol: int, nRow: int):
        if x == nCol // 2 + 1 or y == nRow // 2 + 1:
            return True
        return False

    @staticmethod
    def _FGHSet(
        nCol: int, nRow: int, D: float, mu: float = 0.5, E: float = 5000, L: float = 0
    ):
        Kx = TFF._get_Wavefunction_in_FS(nCol, D)
        Ky = TFF._get_Wavefunction_in_FS(nRow, D)
        H = np.zeros([2 * nCol * nRow, 2 * nCol * nRow], dtype=np.complex)
        I = np.identity(2, dtype=np.complex) * L * L
        for i in range(nRow):
            for j in range(nCol):
                o = (i * nCol + j) * 2
                flag = TFF._is_edge(j, i, nCol, nRow)
                G = TFF._calc_Green(Kx[j], Ky[i], flag, mu, E)
                # Gtinv = np.linalg.inv(G.T)
                # G1 = G.T @ G + H
                H[o : o + 2, o : o + 2] = G
        F = np.identity(2 * nCol * nRow)
        G = np.ones(2 * nCol * nRow)
        return F, G, H

    def confirm(self):
        return TFF(super().confirm())

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        return super().rearrange_by_coordinate(target)

    def draw(self, scale: int = None, save_img: bool = False, name: str = None):
        super().draw(scale=scale, save_img=save_img, name=name)

