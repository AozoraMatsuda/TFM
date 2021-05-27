import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
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
            data, columns=["x", "y", "vx", "vy", "m"], dtype=np.float64
        ).dropna()
        return DPF(df).confirm()

    @classmethod
    def PIV(cls, path: str, wsize: int = 32, overlap: int = 16, pixel: float = 0.090):
        for path1, path2 in zip(path[:-1], path[1:]):
            img1 = cv2.imread(path1, 0)
            img2 = cv2.imread(path2, 0)
            coordinates = []
            h1, w1 = img1.shape
            h2, w2 = img2.shape

            if h1 != h2 or w1 != w2:
                assert ValueError("Align image size")

            w_st = int(w1 / (wsize - overlap))
            h_st = int(h1 / (wsize - overlap))

            for i in range(h_st - 1):
                for j in range(w_st - 1):
                    p_h1 = i * (wsize - overlap)
                    p_h2 = p_h1 + wsize
                    p_w1 = j * (wsize - overlap)
                    p_w2 = p_w1 + wsize

                    template = img1[p_h1:p_h2, p_w1:p_w2]

                    method = cv2.TM_CCOEFF_NORMED
                    res = cv2.matchTemplate(img2, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    before_w = p_w1 + (p_w2 - p_w1) / 2
                    before_h = p_h1 + (p_h2 - p_h1) / 2

                    after_w = max_loc[0] + wsize / 2
                    after_h = max_loc[1] + wsize / 2
                    # print(before_w,before_h, after_w,after_h)

                    dx = (after_w - before_w) * pixel
                    dy = (after_h - before_h) * pixel

                    coordinates.append([before_w, before_h, dx, dy])

            df = pd.DataFrame(coordinates)
            df.columns = ["x", "y", "vx", "vy"]
            df["m"] = (df["vx"] ** 2 + df["vy"] ** 2).pow(1 / 2)
            df = DPF(df).confirm()
            return df

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
        return DPF(super().confirm())

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        return super().rearrange_by_coordinate(target)

    def draw(self, scale: int = None, save_img: bool = False, name: str = None):
        super().draw(scale=scale, save_img=save_img, name=name)
