#%%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.core.numeric import identity
from numpy.lib.npyio import save
from pandas.util.testing import assert_frame_equal
from pykalman import KalmanFilter
from sklearn.metrics import r2_score

from TFM import DPF, TFF

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


def Laplacian(lap, Tv):
    lap[1:-1, 1:-1] = (
        -4 * Tv[1:-1, 1:-1]
        + Tv[:-2, 1:-1]
        + Tv[2:, 1:-1]
        + Tv[1:-1, :-2]
        + Tv[1:-1, 2:]
    )
    lap[0, 0] = -2 * Tv[0, 0] + Tv[0, 1] + Tv[1, 0]
    lap[0, -1] = -2 * Tv[0, -1] + Tv[0, -2] + Tv[1, -1]
    lap[-1, 0] = -2 * Tv[-1, 0] + Tv[-2, 0] + Tv[-1, 1]
    lap[-1, -1] = -2 * Tv[-1, -1] + Tv[-1, -2] + Tv[-2, -1]
    lap[1:-1, 0] = -3 * Tv[1:-1, 0] + Tv[:-2, 0] + Tv[2:, 0] + Tv[1:-1, 1]
    lap[1:-1, -1] = -3 * Tv[1:-1, -1] + Tv[:-2, -1] + Tv[2:, -1] + Tv[1:-1, -2]
    lap[0, 1:-1] = -3 * Tv[0, 1:-1] + Tv[0, :-2] + Tv[0, 2:] + Tv[1, 1:-1]
    lap[-1, 1:-1] = -3 * Tv[-1, 1:-1] + Tv[-1, :-2] + Tv[-1, 2:] + Tv[-2, 1:-1]


def update(Wx, Wy, mode: str = None):
    # Euler法によるcGL方程式の
    if mode == "cGL":
        _calc_Laplacian(lapx, Wx)  # Laplacian of Tx
        _calc_Laplacian(lapy, Wy)  # Laplacian of Ty
        W2[:] = Wx[:] * Wx[:] + Wy[:] * Wy[:]  # |T|^2
        kWx[:] = Wx[:] - W2[:] * (Wx[:] - b * Wy[:]) + D * (lapx[:] - a * lapy[:])
        kWy[:] = Wy[:] - W2[:] * (b * Wx[:] + Wy[:]) + D * (a * lapx[:] + lapy[:])
        Wx[:, :] = Wx[:, :] + dt * kWx[:, :] + np.random.normal(0, ni, (nCol, nRow))
        Wy[:, :] = Wy[:, :] + dt * kWy[:, :] + np.random.normal(0, ni, (nCol, nRow))


Wx[:] = np.random.normal(0, 1, (nCol, nRow))
Wy[:] = np.random.normal(0, 1, (nCol, nRow))


fig = plt.figure(figsize=(5, 5))
plt.xlim([0, nCol * dx])  # 描くXの範囲
plt.ylim([0, nRow * dx])  # 描くyの範囲
X, Y = np.meshgrid(np.arange(0, nCol * dx, dx), np.arange(0, nRow * dx, dx))  # メッシュ生成
S = 4  ##  表示のためのメッシュ間隔
OUTSTEP = 500  ## step毎に表示


for t in range(TIMESTEP):
    update(Wx, Wy)

    if t % OUTSTEP == 0:
        # plt.imshow(Tx)
        plt.clf()
        plt.title("t= %.2lf" % (t * dt,))
        plt.imshow(Wx)
        plt.quiver(
            X[::S, ::S],
            Y[::S, ::S],
            Wx[::S, ::S],
            Wy[::S, ::S],
            color="red",
            angles="xy",
            scale_units="xy",
            scale=0.2,
        )
        # plt.quiver(X[::S,::S],Y[::S,::S],-Wy[::S,::S],Wx[::S,::S],color='red',angles='xy',scale_units='xy', scale=.2)
        plt.pause(1)
    # %%
    def _update(
        Wx: np.ndarray,
        Wy: np.ndarray,
        mode: str = None,
        info: dict = None,
        noise_flag: int = 1,
    ):
        nCol, nRow = Wx.shape
        # Euler法によるcGL方程式の
        if mode == "cGL":
            a = info["a"]
            b = info["b"]
            dx = info["dx"]
            dt = info["dt"]
            D = 2.0 / dx / dx
            ni = np.sqrt(0.4 * dt)
            lapx = _calc_Laplacian(Wx)  # Laplacian of Tx
            lapy = _calc_Laplacian(Wy)  # Laplacian of Ty
            W2 = Wx * Wx + Wy * Wy  # |T|^2
            kWx = Wx - W2 * (Wx - b * Wy) + D * (lapx - a * lapy)
            kWy = Wy - W2 * (b * Wx + Wy) + D * (a * lapx + lapy)
            Wx = Wx + dt * kWx + np.random.normal(0, ni, (nCol, nRow)) * noise_flag
            Wy = Wy + dt * kWy + np.random.normal(0, ni, (nCol, nRow)) * noise_flag

    def generate_dpf(
        nCol: int = 200, nRow: int = 200, mode: str = "cGL", info: dict = None
    ):
        if mode == "cGL":
            dx = info["dx"]
            dt = info["dt"]
            TIMESTEP = 1000
        Wx = np.random.normal(0, 1, (nCol, nRow))
        Wy = np.random.normal(0, 1, (nCol, nRow))
        X, Y = np.meshgrid(
            np.arange(0, nCol * dx, dx), np.arange(0, nRow * dx, dx)
        )  # メッシュ生成
        res = []
        for t in range(TIMESTEP):
            _update(Wx=Wx, Wy=Wy, mode="cGL", info=info)

            df = TFF(
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

    def _calc_Laplacian(Tv: np.ndarray):
        lap = np.zeros(Tv.shape)
        lap[1:-1, 1:-1] = (
            -4 * Tv[1:-1, 1:-1]
            + Tv[:-2, 1:-1]
            + Tv[2:, 1:-1]
            + Tv[1:-1, :-2]
            + Tv[1:-1, 2:]
        )
        lap[0, 0] = -2 * Tv[0, 0] + Tv[0, 1] + Tv[1, 0]
        lap[0, -1] = -2 * Tv[0, -1] + Tv[0, -2] + Tv[1, -1]
        lap[-1, 0] = -2 * Tv[-1, 0] + Tv[-2, 0] + Tv[-1, 1]
        lap[-1, -1] = -2 * Tv[-1, -1] + Tv[-1, -2] + Tv[-2, -1]
        lap[1:-1, 0] = -3 * Tv[1:-1, 0] + Tv[:-2, 0] + Tv[2:, 0] + Tv[1:-1, 1]
        lap[1:-1, -1] = -3 * Tv[1:-1, -1] + Tv[:-2, -1] + Tv[2:, -1] + Tv[1:-1, -2]
        lap[0, 1:-1] = -3 * Tv[0, 1:-1] + Tv[0, :-2] + Tv[0, 2:] + Tv[1, 1:-1]
        lap[-1, 1:-1] = -3 * Tv[-1, 1:-1] + Tv[-1, :-2] + Tv[-1, 2:] + Tv[-2, 1:-1]
        return lap


# %%da
# %%
info = {"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.01}
# %%
sym_data = TFF.generate_fields(
    nCol=200,
    nRow=200,
    size=500,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.01},
)
#%%
noise = 0.2
data_noise = []
for df in sym_data:
    ndf = df.copy()
    nCol, nRow, _ = df.get_Dimensions()
    mx = df.loc[:, "vx"].abs().mean() * noise
    my = df.loc[:, "vy"].abs().mean() * noise
    ndf.loc[:, "vx"] += np.random.normal(0, mx, nCol * nRow)
    ndf.loc[:, "vy"] += np.random.normal(0, my, nCol * nRow)
    data_noise.append(ndf)
#%%
sym_train = [x.inv_fftc(noise_flag=False) for x in sym_data]
# train_noise = [x.inv_fftc(noise_ratio=0.05) for x in data_noise]
#%%
result_est = TFF.kalman_FFTC(data=sym_train[:200],)


#%%
with open("tfm.tff", "rb") as fl:
    tff = pickle.load(fl)
#%%
for i in range(200):
    if i % 20 == 0:
        sym_data[i + 1].draw()
        result_est[i].draw()
        print("###################")

#%%
train = [x.convert_to_dpf(noise_flag=False) for x in data]
#%%

result = TFF.kalman_FFTC(data=train[:101])
#%%
for i in range(100):
    if i % 20 == 0:
        result[i].draw()
        sym_data[i].draw()

# %%
rel_error_0 = (
    result[0].loc[:, ["vx", "vy"]] - data[1].loc[:, ["vx", "vy"]]
).abs() / data[1].loc[:, ["vx", "vy"]].abs()
rel_error_50 = (
    result[50].loc[:, ["vx", "vy"]] - data[51].loc[:, ["vx", "vy"]]
).abs() / data[51].loc[:, ["vx", "vy"]].abs()
rel_error_99 = (
    result[99].loc[:, ["vx", "vy"]] - data[100].loc[:, ["vx", "vy"]]
).abs() / data[100].loc[:, ["vx", "vy"]].abs()

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
    nCol=150,
    nRow=150,
    size=2000,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 1e-20},
)
#%%
ratio = 0.4
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
result_d0 = TFF.kalman_FFTC(data=train_noise[0:1000])
#%%
result_d1 = TFF.kalman_FFTC(data=train_exd[0:150], use_em=True, mode=1)

#%%
for i in range(100):
    if i % 20 == 0:
        print("#################")
        sym_data[3*i].draw(save_img=True, name=f'sym_3_{i}')
        result_d0[i].draw(save_img=True, name=f'd0_3_{i}')
        result_d1[i].draw(save_img=True, name=f'd1_3_{i}')
# %%
sym_data = TFF.generate_fields(
    nCol=100,
    nRow=100,
    size=2000,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.000001},
)
# %%
train = train_noise[::10]
T = 10
train_exd = [train[0]]
for i in range(len(train) - 1):
    df0, df1 = train[i], train[i + 1]
    for i in range(1, T):
        df = df0.copy()
        df["vx"] = df0["vx"] * (1 - i / T) + df1["vx"] * (i / T)
        df["vy"] = df0["vy"] * (1 - i / T) + df1["vy"] * (i / T)
        df["m"] = (df["vx"].pow(2) + df["vy"].pow(2)).pow(1 / 2)
        train_exd.append(DPF(df.copy()))
    train_exd.append(DPF(df1))
# %%
for i in range(200):
    if i % 20 == 0:
        train_noise[i].draw()
        train_exd[i].draw()
# %%
exd_0 = []
norm_exd_0 = []
exd_1 = []
norm_exd_1 = []
cmp = []
for i in range(500):
    exd_0.append(
        r2_score(
            sym_data[1 * (i+1)].loc[:, ["vx", "vy"]].values,
            result_d0[i].loc[:, ["vx", "vy"]].values,
        )
    )
    cmp.append(
        r2_score(
            sym_data[1 * (i+1)].loc[:, ["vx", "vy"]].values,
            train_noise[1 * (i+1)].fftc().loc[:, ["vx", "vy"]].values,
        )
    )
    # df = (sym_data[1*(i+1)] - result_d0[i]).loc[:, ["vx", "vy"]]
    # m = np.sqrt((df["vx"].pow(2) + df["vy"].pow(2)).sum())
    # norm_exd_0.append(m)

    # exd_1.append(
    #     r2_score(
    #         sym_data[1 * (i+1) ].loc[:, ["vx", "vy"]].values,
    #         result_d1[i].loc[:, ["vx", "vy"]].values,
    #     )
    # )
    # df = (sym_data[1*(i+1)] - result_d1[i]).loc[:, ["vx", "vy"]]
    # m = np.sqrt((df["vx"].pow(2) + df["vy"].pow(2)).sum())
    # norm_exd_1.append(m)

    # cmp.append(
    #     r2_score(
    #         sym_data[1 * (i+1) ].loc[:, ["vx", "vy"]].values,
    #         train_noise[1 * (i+1) ].fftc().loc[:, ["vx", "vy"]].values,
    #     )
    # )
# %%
plt.plot(exd_0, label='d0')
plt.plot(cmp, label='cmp')
plt.legend()

# %%
