#%%
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.core.numeric import identity
from numpy.lib.npyio import save
from pandas.util.testing import assert_frame_equal, assert_series_equal
from pykalman import KalmanFilter
from sklearn.metrics import r2_score
from scipy.sparse import csr_matrix
from TFM import DPF, TFF, SparseKalman

# %%
path_A = "./example/PIV_A001_stack.txt"
piv_A001 = DPF.load_DPF(path_A)
path_B = "./example/PIV_B001_stack.txt"
piv_B001 = DPF.load_DPF(path_B)

# %%
piv_A001.draw()
piv_B001.draw()

#%%
res_A = piv_A001.fftc()
res_A.draw()
expected_A = TFF.load_TFF(path="./example/Traction_PIV_A001_stack_L_0.txt")
expected_A.draw()
assert_frame_equal(res_A, expected_A)

#%%
res_B = piv_B001.fftc()
res_B.draw()
expected_B = TFF.load_TFF(path="./example/Traction_PIV_B001_stack.txt")
expected_B.draw()
assert_frame_equal(res_B, expected_B)

#%%
res = piv_A001.fftc(L=0.005)
res.draw()

# %%
path = ["~/Desktop/TFM/Aall/A001a.tif", "~/Desktop/TFM/Aall/A001b.tif"]
DPF.PIV(path, save_path="~/Desktop")
# %%
path = ["~/Desktop/TFM/Aall/A001a.tif", "~/Desktop/TFM/Aall/A001b.tif"]
df = DPF.PIV(path, save_path="~/Desktop")
path_B = "../PIV/PIV_B001_stack.txt"
piv_B001 = DPF.load_DPF(path_B)

# %%
ls = ["./img1.png", "./img2.png"]
for s in ls:
    df = DPF.load_DPF(s)
    ls.append(DPF(df))
#%%
initial_tff = TFF.FFTC(ls[0])
res = TFF.kalman_FFTC(ls)
#%%
initial_tff.draw()
res.draw()
import matplotlib.pyplot as plt

# %%
import numpy as np

nCol = 200  # meshsize_x
nRow = 200  # meshsize_y
dx = 1.0  # mesh間隔
dt = 0.01  # discretized timestep
ni = np.sqrt(0.4 * dt)  # ノイズの大きさ
TIMESTEP = 10000

Wx = np.zeros((nCol, nRow), float)
Wy = np.zeros((nCol, nRow), float)
kWx = np.zeros((nCol, nRow), float)
kWy = np.zeros((nCol, nRow), float)
lapx = np.zeros((nCol, nRow), float)
lapy = np.zeros((nCol, nRow), float)
W2 = np.zeros((nCol, nRow), float)

a = 0.5
b = 0.1
D = 2.0 / dx / dx


Wx[:] = np.random.normal(0, 1, (nCol, nRow))
Wy[:] = np.random.normal(0, 1, (nCol, nRow))


fig = plt.figure(figsize=(5, 5))
plt.xlim([0, nCol * dx])  # 描くXの範囲
plt.ylim([0, nRow * dx])  # 描くyの範囲
X, Y = np.meshgrid(np.arange(0, nCol * dx, dx), np.arange(0, nRow * dx, dx))  # メッシュ生成
S = 4  ##  表示のためのメッシュ間隔
OUTSTEP = 500  ## step毎に表示


# %%da
# %%
info = {"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.01}
# %%
sym_data = TFF.generate_fields(
    nCol=200,
    nRow=200,
    size=200,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.01},
)
#%%
data_noise = []
for df in sym_data:
    ndf = df.copy()
    nCol, nRow, _ = df.get_Dimensions()
    mx = df.loc[:, "vx"].abs().mean() * 0.05
    my = df.loc[:, "vy"].abs().mean() * 0.05
    ndf.loc[:, "vx"] += np.random.normal(0, mx, nCol * nRow)
    ndf.loc[:, "vy"] += np.random.normal(0, my, nCol * nRow)
    data_noise.append(ndf)
#%%
sym_train = [x.inv_fftc(noise_flag=False) for x in sym_data]
# train_noise = [x.inv_fftc(noise_ratio=0.05) for x in data_noise]
#%%
result_est = TFF.kalman_FFTC(data=sym_train[:100], use_em=False, mode=0)


#%%
with open("tfm.tff", "rb") as fl:
    tff = pickle.load(fl)

for i in range(200):
    if i % 20 == 0:
        sym_data[i + 1].draw()
        result_est[i].draw()
        print("###################")


#%%
with open("piv.dpf", "rb") as fl:
    data = pickle.load(fl)
train = []
for i in range(len(data)):
    df = data[i]
    mask = (1400 < df["x"]) & (1400 < df["y"]) & (df["x"] < 1500) & (df["y"] < 1500)
    train.append(DPF(df[mask].copy()))
# %%
T = 10
train_exd = []
for i in range(len(train) - 1):
    df0, df1 = train[i], train[i + 1]
    train_exd.append(DPF(df0.copy()))
    for i in range(1, T):
        df = df0.copy()
        df["vx"] = df0["vx"] * (1 - i / T) + df1["vx"] * (i / T)
        df["vy"] = df0["vy"] * (1 - i / T) + df1["vy"] * (i / T)
        df["m"] = (df["vx"].pow(2) + df["vy"].pow(2)).pow(1 / 2)
        train_exd.append(DPF(df.copy()))
    train_exd.append(DPF(df1))
#%%
result_est = TFF.kalman_FFTC(
    data=train, use_em=True, mu=0.5, E=15000.0, pixel=6.4e-7, mode=1
)

#%%
with open("tfm.tff", "rb") as fl:
    tff = pickle.load(fl)

for i in range(200):

    if i % 20 == 0:
        df = tff[i]
        mask = (1400 < df["x"]) & (1400 < df["y"]) & (df["x"] < 1460) & (df["y"] < 1460)
        df[mask].draw()
        # train[i].fftc(mu=0.5, E=15000, pixel=6.4E-7,L=8.0E-11).draw()
        result_est[i].draw()
        print("#######")
# %%
sym_data = TFF.generate_fields(
    nCol=200,
    nRow=200,
    size=500,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 1e-20},
)
#%%
ratio = 0.2
data_noise = []
for df in sym_data:
    ndf = df.copy()
    nCol, nRow, _ = df.get_Dimensions()
    mx = df.loc[:, "vx"].abs().mean() * ratio
    my = df.loc[:, "vy"].abs().mean() * ratio
    ndf.loc[:, "vx"] += np.random.normal(0, mx, nCol * nRow)
    ndf.loc[:, "vy"] += np.random.normal(0, my, nCol * nRow)
    data_noise.append(ndf)
#%%
# sym_train = [x.inv_fftc(noise_flag=False) for x in sym_data]
train_noise = [x.inv_fftc(noise_ratio=ratio) for x in data_noise]
#%%
result_d0 = TFF.kalman_FFTC(data=train_noise[0:400], mode=0, noise=1, use_em=False)
# %%
for i in range(100):
    if i % 20 == 0:
        print("#################")
        sym_data[i].draw()
        result_d0[i].draw()
# %%
exd_0 = []
cmp = []
for i in range(300):
    exd_0.append(
        r2_score(
            sym_data[i + 1].loc[:, ["vx", "vy"]].values,
            result_d0[i].loc[:, ["vx", "vy"]].values,
        )
    )
    cmp.append(
        r2_score(
            sym_data[i + 1].loc[:, ["vx", "vy"]].values,
            train_noise[i + 1].fftc().loc[:, ["vx", "vy"]].values,
        )
    )

# %%
plt.plot(exd_0, label="d0")
plt.plot(cmp, label="cmp")
plt.legend()

# %%
