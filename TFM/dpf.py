import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import imagej
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
    def PIV(
        cls,
        path: str,
        piv1: int = 64,
        sw1: int = 128,
        piv2: int = 32,
        sw2: int = 64,
        piv3: int = 16,
        sw3: int = 32,
        correlation: float = 0.60,
        save_path: str = None,
    ) -> list:
        print("START LOADIND FIJI....")
        ij = imagej.init("/Applications/Fiji.app")
        print("DONE!")
        cnt = 1
        res = []
        for path1, path2 in zip(path[:-1], path[1:]):
            path = save_path + "/" + str(cnt) + ".txt"
            ij.py.run_macro(f"""open({path1});""")
            ij.py.run_macro(f"""open({path2});""")
            ij.py.run_macro("""run("Images to Stack", "name=Stack title=[] use");""")
            ij.py.run_macro(
                """
run("iterative PIV(Basic)...", f"{piv1}=32 {sw1}=64 {piv2}=16 {sw2}=32 {piv3}=8 {sw3}=16 {correlation}=0.60 what=[Accept this PIV and output] noise=0.20 threshold=5 c1=3 c2=1 save=[{save_path}]");
"""
            )
            df = DPF.load_DPF(path)
            res.append(df)
        return res

    def get_Dimensions(self) -> list:
        return super().get_Dimensions()

    def confirm(self):
        return DPF(super().confirm())

    def rearrange_by_coordinate(self, target: str) -> pd.DataFrame:
        return super().rearrange_by_coordinate(target)

    def draw(self, scale: int = None, save_img: bool = False, name: str = None):
        super().draw(scale=scale, save_img=save_img, name=name)
