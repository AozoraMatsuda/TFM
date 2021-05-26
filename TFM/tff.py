from os import stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            data, columns=["x", "y", "f_x", "f_y", "m"], dtype=np.float64
        ).dropna()
        return TFF(df)

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
        disX = disXY.rearrange_for_coordinate("d_x")
        disX *= pixel
        disY = disXY.rearrange_for_coordinate("d_y")
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
        for j in range(len(Ky)):
            for i in range(len(Kx)):
                k = np.sqrt(Kx[i] * Kx[i] + Ky[j] * Ky[j])
                if i == nCol // 2 + 1 or j == nRow // 2 + 1:
                    G[0, 1] = 0
                    G[1, 0] = 0
                elif i != 0 or j != 0:
                    gg = -mu * Kx[i] * Ky[j]
                    G[0, 1] = gg
                    G[1, 0] = gg
                G0 = 2 * (1 + mu) / (E * pow(k, 3))
                G[0, 0] = (1 - mu) * (k * k) + mu * (Ky[j] * Ky[j])
                G[1, 1] = (1 - mu) * (k * k) + mu * (Kx[i] * Kx[i])
                G *= G0

                Gt = G.T
                G1 = Gt @ G
                G1 += H
                Ginv = np.linalg.inv(G1)

                dd = np.array([disXCF[j, i], disYCF[j, i]])
                GtU = Gt @ dd
                TXY = Ginv @ GtU
                TractionXF[j, i] = TXY[0]
                TractionYF[j, i] = TXY[1]
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
                "f_x": TractionXR,
                "f_y": TractionYR,
                "m": magnitude,
            }
        )
        return TFF(df)

    @staticmethod
    def _get_Wavefunction_in_FS(num: int, D: int) -> np.array:
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

    def rearrange_for_coordinate(self, target: str) -> pd.DataFrame:
        return super().rearrange_for_coordinate(target)

    def draw(self, scale: int = None, save_img: bool = False, name: str = None):
        super().draw(scale=scale, save_img=save_img, name=name)

