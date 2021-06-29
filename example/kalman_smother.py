#%%
import sys

sys.path.append("../")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.util.testing import assert_frame_equal
from TFM import DPF, TFF
from sklearn.metrics import r2_score

# %%
# %%
# 　元となるTFFのリストを生成
data = TFF.generate_fields(
    nCol=150,
    nRow=150,
    size=210,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.001},
)
# %%
#%%
# 　ノイズの追加
data_noise = []
noise = 0.0
for df in data:
    ndf = df.copy()
    nCol, nRow, _ = df.get_Dimensions()
    mx = df.loc[:, "vx"].abs().mean() * noise
    my = df.loc[:, "vy"].abs().mean() * noise
    ndf.loc[:, "vx"] += np.random.normal(0, mx, nCol * nRow)
    ndf.loc[:, "vy"] += np.random.normal(0, my, nCol * nRow)
    data_noise.append(ndf)

train_noise = [x.inv_fftc(noise_ratio=noise) for x in data_noise]
#%%
result = TFF.kalman_FFTC(data=train_noise[:201])

#%%
fig = plt.figure()
k_r2_x = []
k_r2_y = []
f_r2_x = []
f_r2_y = []
for i in range(200):
    res = result[i]
    cp = train_noise[i + 1].fftc()
    ept = data[i + 1]
    k_r2_x.append(r2_score(ept.loc[:, "vx"].values, res.loc[:, "vx"].values))
    k_r2_y.append(r2_score(ept.loc[:, "vy"].values, res.loc[:, "vy"].values,))
    f_r2_x.append(r2_score(ept.loc[:, "vx"].values, cp.loc[:, "vx"].values))
    f_r2_y.append(r2_score(ept.loc[:, "vy"].values, cp.loc[:, "vy"].values))

    if i % 40 == 0:
        ept.draw(figsize=(50, 50), save_img=True, name=str(i) + f"_expected_{noise}")
        cp.draw(figsize=(50, 50), save_img=True, name=str(i) + f"_fftc_{noise}")
        result[i].draw(
            figsize=(50, 50), save_img=True, name=str(i) + f"_kalman_{noise}"
        )

        fftc_dif = (ept - cp).loc[:, ["vx", "vy"]]
        norm_fftc_error = (fftc_dif["vx"] ** 2 + fftc_dif["vy"] ** 2).pow(1 / 2)

        kalman_dif = (ept - result[i]).loc[:, ["vx", "vy"]]
        norm_kalman_error = (kalman_dif["vx"] ** 2 + kalman_dif["vy"] ** 2).pow(1 / 2)

        plt.hist(
            [norm_fftc_error.values, norm_kalman_error.values],
            cumulative=True,
            bins=10,
            label=["FFTC", "KS-FFTC"],
        )
        plt.legend()
        plt.savefig(f"kalman_{i}_{noise}.png")
        plt.show()

# %%
# %%
plt.hist([k_r2_y, f_r2_y], bins=10, label=["KS-FFTC", "FFTC"])
plt.legend()
plt.savefig(f"r2_y_{noise}.png")
plt.show()
# %%
plt.plot(k_r2_x)
plt.savefig(f"k2_x_{noise}.png")
plt.show()
# %%
plt.plot(k_r2_y)
plt.savefig(f"k2_y_{noise}.png")
plt.show()
# %%
plt.plot(f_r2_x)
plt.savefig(f"f2_x_{noise}.png")
plt.show()
# %%
plt.plot(f_r2_y)
plt.savefig(f"f2_y_{noise}.png")
plt.show()
# %%