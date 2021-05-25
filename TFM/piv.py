import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PIV(pd.DataFrame):
    @property
    def _constrctor(self):
        return PIV

    @classmethod
    def load_PIV(cls, path: str) -> "PIV":
        file = open(path, "r")
        data = file.read()
        data = [s.split()[:5] for s in data.split("\n")]
        df = pd.DataFrame(
            data, columns=["x", "y", "d_x", "d_y", "m"], dtype=np.float64
        ).dropna()
        return PIV(df)
