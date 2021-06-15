import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    def draw(self, scale: int = None, save_img: bool = False, name: str = None):
        df = self.copy()
        fig = plt.figure(figsize=(50, 50), facecolor="#180614", edgecolor="#302833")
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
