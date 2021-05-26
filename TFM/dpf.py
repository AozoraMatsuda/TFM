import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TFM import Vectors


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
            data, columns=["x", "y", "d_x", "d_y", "m"], dtype=np.float64
        ).dropna()
        return DPF(df)

    def get_Dimensions(self) -> list:
        dim = [0] * 3
        dim[2] = self.iloc[1, 0] - self.iloc[0, 0]
        dim[0] = self.iloc[:, 0].nunique()
        dim[1] = self.iloc[:, 1].nunique()
        return dim

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        return super().rearrange_by_coordinate(target)

    def draw(self, scale: int = None, save_img: bool = False, name: str = None):
        super().draw(scale=scale, save_img=save_img, name=name)
