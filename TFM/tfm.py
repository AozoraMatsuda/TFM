import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TFM import PIV


class TFM(pd.DataFrame):
    @property
    def _constructor(self):
        return TFM

    @classmethod
    def load_TFM(cls, path: str) -> "TFM":
        file = open(path, "r")
        data = file.read()
        data = [s.split() for s in data.split("\n")]
        df = pd.DataFrame(
            data, columns=["x", "y", "f_x", "f_y", "m"], dtype=np.float64
        ).dropna()
        return TFM(df)

    @classmethod
    def FFTC(
        cls,
        disXY: "PIV",
        pixel: float = 0.090,
        mu: float = 0.5,
        E: float = 5000,
        L: float = 0,
    ) -> "TFM":

        dim = cls._getDimensions(disXY)

        # get disformation and coordinate
        disX = disXY.loc[:, ["x", "y", "d_x"]] * pixel
        disX = disX.set_index(["y", "x"]).iloc[:, 0].unstack()
        disY = disXY.loc[:, ["x", "y", "d_y"]] * pixel
        disY = disY.set_index(["y", "x"]).iloc[:, 0].unstack()
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
        Kx = (
            (2 * np.pi)
            / (D * nCol)
            * np.hstack(
                [
                    np.arange(0, nCol // 2 + 1, 1),
                    (-1) * np.arange(np.round(nCol / 2) - 1, 0, -1),
                ]
            )
        )
        Ky = (
            (2 * np.pi)
            / (D * nRow)
            * np.hstack(
                [
                    np.arange(0, nRow // 2 + 1, 1),
                    (-1) * np.arange(np.round(nRow / 2) - 1, 0, -1),
                ]
            )
        )

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
        return TFM(df)

    @staticmethod
    def _getDimensions(disXY: "PIV") -> list:
        dim = [0] * 3
        dim[2] = disXY.iloc[1, 0] - disXY.iloc[0, 0]
        dim[0] = disXY.iloc[:, 0].nunique()
        dim[1] = disXY.iloc[:, 1].nunique()
        return dim

    def draw(self, scale: int = 50, save_img: bool = False, name: str = None):
        df = self.copy()
        fig = plt.figure(figsize=(50, 50), facecolor="#180614", edgecolor="#302833")
        ax = fig.add_subplot(111)

        # 背景
        ax.set_facecolor("#180614")
        # 軸ラベルの設定
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("y", fontsize=16)

        # 軸範囲の設定
        m_x, m_y = (
            df["x"].min() - 20,
            df["y"].min() - 20,
        )
        M_x, M_y = df["x"].max() + 20, df["y"].max() + 20
        ax.set_xlim(m_x, M_x)
        ax.set_ylim(m_y, M_y)

        # x軸とy軸
        ax.invert_yaxis()
        ax.axis("off")

        # ベクトル
        X = df.loc[:, "x"]
        Y = df.loc[:, "y"]
        F_X = df.loc[:, "f_x"]
        F_Y = df.loc[:, "f_y"]
        M = df.loc[:"m"]
        ax.quiver(
            X,
            Y,
            F_X,
            F_Y,
            M,
            angles="xy",
            scale_units="xy",
            scale=scale,
            cmap="hot",
            alpha=0.8,
        )
        plt.show()
        if save_img:
            name = name + ".png" if name is not None else "img.png"
            fig.savefig(name, facecolor="#180614", edgecolor="#180614")

