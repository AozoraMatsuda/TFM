import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from TFM.utils import calc_Laplacian


class Vectors(pd.DataFrame):
    def get_Dimensions(self) -> list:
        """
        dim[0] the number of points in x axis
        dim[1] the number of points in y axis
        dim[2] distance between points (pixel)
        """
        dim = [0] * 3
        dim[2] = self.iloc[1, 0] - self.iloc[0, 0]
        dim[0] = self.iloc[:, 0].nunique()
        dim[1] = self.iloc[:, 1].nunique()
        return dim

    def confirm(self):
        df = self.copy()
        df.columns = ["x", "y", "vx", "vy", "m"]
        return df.sort_values(["y", "x"])

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        data = self.loc[:, ["x", "y", target]]
        return data.set_index(["y", "x"]).iloc[:, 0].unstack()

    def draw(
        self, figsize: tuple = (5, 5), scale: int = None, name: str = None,
    ):
        df = self.copy()
        fig = plt.figure(figsize=figsize, facecolor="#180614", edgecolor="#302833")
        ax = fig.add_subplot(111)

        # background
        ax.set_facecolor("#180614")
        # axis label
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("y", fontsize=16)

        # range of axis
        m_x, m_y = df["x"].min() - 20, df["y"].min() - 20
        M_x, M_y = df["x"].max() + 20, df["y"].max() + 20

        width = M_x - m_x
        height = M_y - m_y

        ax.set_xlim(m_x, M_x)
        ax.set_ylim(m_y, M_y)

        # setting axis
        ax.invert_yaxis()
        ax.axis("off")

        # vectors settings
        X = df.iloc[:, 0]
        Y = df.iloc[:, 1]
        F_X = df.iloc[:, 2]
        F_Y = df.iloc[:, 3]
        M = df.iloc[:, 4]

        # auto scale
        scale = (
            max(F_X.abs().max() / (0.05 * width), F_Y.abs().max() / (0.05 * height))
            if scale is None
            else scale
        )
        img = ax.quiver(
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
        if name is not None:
            name = name + ".png"
            fig.savefig(name, facecolor="#180614", edgecolor="#180614")
        return img

    @classmethod
    def generate_fields(
        self, nCol: int, nRow: int, size: int, mode: str = "cGL", info: dict = None
    ) -> list:
        """Return a list of TFF objects which are generated by artificial means

        Args:
            nCol (int): the number of x points
            nRow (int): the number of y points
            size (int): the number of TFF objects
            mode (str, optional): the way how to synthesize traction force field. Defaults to "cGL".
                cGL : https://codeinthehole.com/tutorial/index.html
            info (dict, optional): the parameters for each mode. Defaults to None.

        Returns:
            list: list of TFF objects
        """
        dx = info["dx"]
        Wx = np.random.normal(0, 1, (nCol, nRow))[8::16, 8::16]
        Wy = np.random.normal(0, 1, (nCol, nRow))[8::16, 8::16]
        X, Y = np.meshgrid(
            np.arange(8, nCol * dx, 16), np.arange(8, nRow * dx, 16)
        )  # ??????????????????
        res = []
        for _ in range(size):
            Wx, Wy = self._update(Wx=Wx, Wy=Wy, mode=mode, info=info)

            df = Vectors(
                {
                    "x": X.flatten(),
                    "y": Y.flatten(),
                    "vx": Wx.flatten(),
                    "vy": Wy.flatten(),
                    "m": np.sqrt(Wx ** 2 + Wy ** 2).flatten(),
                }
            )
            res.append(df)
        return res

    @staticmethod
    def _update(
        Wx: np.ndarray,
        Wy: np.ndarray,
        mode: str = None,
        info: dict = None,
        noise_flag: int = 1,
    ):
        nRow, nCol = Wx.shape
        # Euler????????????cGL?????????

        if mode == "cGL":
            a = info["a"]
            b = info["b"]
            dx = info["dx"]
            dt = info["dt"]
            D = 2.0 / dx / dx
            ni = np.sqrt(0.4 * dt)
            lapx = calc_Laplacian(Wx)  # Laplacian of Tx
            lapy = calc_Laplacian(Wy)  # Laplacian of Ty
            W2 = Wx * Wx + Wy * Wy  # |T|^2
            kWx = Wx - W2 * (Wx - b * Wy) + D * (lapx - a * lapy)
            kWy = Wy - W2 * (b * Wx + Wy) + D * (a * lapx + lapy)
            Wx = Wx + dt * kWx + np.random.normal(0, ni, (nCol, nRow)) * noise_flag
            Wy = Wy + dt * kWy + np.random.normal(0, ni, (nCol, nRow)) * noise_flag
        return Wx, Wy
