#%%
import sys

sys.path.append("../")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.util.testing import assert_frame_equal
from TFM import DPF, TFF
from sklearn.metrics import r2_score
import pickle

np.random.seed(0)
# %%
# %%
# 　元となるTFFのリストを生成
np.random.seed(0)
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
np.random.seed(0)
data_noise = []
noise = 1.0
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
np.random.seed(0)
fftc = []
for df in train_noise:
    fftc.append(df.fftc())
#%%
result = TFF.kalman_FFTC(data=train_noise[:201])

#%%
fig = plt.figure()
k_r2_x = []
k_r2_y = []
f_r2_x = []
f_r2_y = []
k_r2 = []
f_r2 = []
k_norms = []
f_norms = []
for i in range(200):
    res = result[i]
    cp = fftc[i + 1]
    ept = data[i + 1]
    k_r2.append(
        r2_score(ept.loc[:, ["vx", "vy"]].values, res.loc[:, ["vx", "vy"]].values)
    )
    # k_r2_y.append(r2_score(ept.loc[:, ["vx","vy"]].values, res.loc[:, ["vx","vy"]].values,))
    f_r2.append(
        r2_score(ept.loc[:, ["vx", "vy"]].values, cp.loc[:, ["vx", "vy"]].values)
    )
    # f_r2_y.append(r2_score(ept.loc[:, ["vx","vy"]].values, cp.loc[:, ["vx","vy"]].values))
    fftc_dif = (ept - cp).loc[:, ["vx", "vy"]]
    norm_fftc_error = (fftc_dif["vx"] ** 2 + fftc_dif["vy"] ** 2).pow(1 / 2).sum()
    f_norms.append(norm_fftc_error)

    kalman_dif = (ept - res).loc[:, ["vx", "vy"]]
    norm_kalman_error = (kalman_dif["vx"] ** 2 + kalman_dif["vy"] ** 2).pow(1 / 2).sum()
    k_norms.append(norm_kalman_error)

    if i % 40 == 0:
        ept.draw(figsize=(50, 50), save_img=True, name=str(i) + f"_expected_{noise}")
        cp.draw(figsize=(50, 50), save_img=True, name=str(i) + f"_fftc_{noise}")
        res.draw(figsize=(50, 50), save_img=True, name=str(i) + f"_kalman_{noise}")

    # plt.hist(
    #     [norm_fftc_error.values, norm_kalman_error.values],
    #     cumulative=True,
    #     bins=10,
    #     label=["FFTC", "KS-FFTC"],
    # )
    # plt.legend()
    # plt.savefig(f"kalman_{i}_{noise}.png")
    # plt.show()

# # %%
# # %%
# plt.plot(k_r2_x)
# plt.savefig(f"k2_x_{noise}.png")
# plt.show()
# # %%
# plt.plot(k_r2_y)
# plt.savefig(f"k2_y_{noise}.png")
# plt.show()
# # %%
# plt.plot(f_r2_x)
# plt.savefig(f"f2_x_{noise}.png")
# plt.show()
# # %%
# plt.plot(f_r2_y)
# plt.savefig(f"f2_y_{noise}.png")
# plt.show()
# %%
fig = plt.figure(figsize=(10, 8))
plt.plot(k_r2, marker="o", markersize=3, label="KS")
plt.plot(f_r2, marker="o", markersize=3, label="FFTC")
plt.legend()
plt.savefig(f"r2_{noise}.png")
plt.show()
# %%
fig = plt.figure(figsize=(10, 8))
plt.plot(k_norms, marker="o", markersize=3, label="KS")
plt.plot(f_norms, marker="o", markersize=3, label="FFTC")
plt.legend()
plt.savefig(f"norm_{noise}.png")
plt.show()
# %%
