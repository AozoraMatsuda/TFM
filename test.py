#%%
from numpy.core.numeric import identity
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.util.testing import assert_frame_equal
from TFM import DPF, TFF
from pykalman import KalmanFilter

# %%
path_A = "../PIV/PIV_A001_stack.txt"
piv_A001 = DPF.load_DPF(path_A)
path_B = "../PIV/PIV_B001_stack.txt"
piv_B001 = DPF.load_DPF(path_B)

# %%
piv_A001.draw()
piv_B001.draw()

#%%
res_A = piv_A001.fftc()
res_A.draw()
expected_A = TFF.load_TFF(path="../traction_force/Traction_PIV_A001_stack_L_0.txt")
expected_A.draw()
assert_frame_equal(res_A, expected_A)

#%%
res_B = piv_B001.fftc()
res_B.draw()
expected_B = TFF.load_TFF(path="../traction_force/Traction_PIV_B001_stack.txt")
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
import imagej

# %%
ij = imagej.init()
# %%
ij = imagej.init("sc.fiji:fiji")
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
# %%
import numpy as np
import matplotlib.pyplot as plt

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
data = TFF.generate_fields(
    nCol=200,
    nRow=200,
    size=200,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.01},
)
#%%
data_noise = []
for df in data:
    ndf = df.copy()
    nCol, nRow, _ = df.get_Dimensions()
    mx = df.loc[:, "vx"].abs().mean() * 0.05
    my = df.loc[:, "vy"].abs().mean() * 0.05
    ndf.loc[:, "vx"] += np.random.normal(0, mx, nCol * nRow)
    ndf.loc[:, "vy"] += np.random.normal(0, my, nCol * nRow)
    data_noise.append(ndf)
#%%
train = [x.inv_fftc(noise_flag=False) for x in data]
train_noise = [x.inv_fftc(noise_ratio=0.05) for x in data_noise]
#%%
result = TFF.kalman_FFTC(data=train[:101])
#%%
for i in range(100):
    if i % 20 == 0:
        data[i + 1].draw()
        result[i].draw(save_img=True, name=str(i) + "_no_noise")

#%%
train = [x.convert_to_dpf(noise_flag=False) for x in data]
#%%
result = TFF.kalman_FFTC(data=train[:101])
#%%
for i in range(100):
    if i % 20 == 0:
        result[i].draw(save_img=True, name=str(i) + "_result_no_noise")
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

