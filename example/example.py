#%%
import sys

sys.path.append("../")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.util.testing import assert_frame_equal
from TFM import DPF, TFF

# %%
path_A = "./PIV_A001_stack.txt"
piv_A001 = DPF.load_DPF(path_A)
path_B = "./PIV_B001_stack.txt"
piv_B001 = DPF.load_DPF(path_B)

# %%
piv_A001.draw()
piv_B001.draw()

#%%
res_A = piv_A001.fftc()
res_A.draw()
expected_A = TFF.load_TFF(path="./Traction_PIV_A001_stack_L_0.txt")
expected_A.draw()
assert_frame_equal(res_A, expected_A)

#%%
res_B = piv_B001.fftc()
res_B.draw()
expected_B = TFF.load_TFF(path="./Traction_PIV_B001_stack.txt")
expected_B.draw()
assert_frame_equal(res_B, expected_B)

#%%
res = piv_A001.fftc(L=0.005)
res.draw()
# %%
# 　元となるTFFのリストを生成
data = TFF.generate_fields(
    nCol=200,
    nRow=200,
    size=200,
    mode="cGL",
    info={"a": 0.5, "b": 0.1, "dx": 1, "dt": 0.01},
)
#%%
# 　ノイズの追加
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
# 　DPFへの変換
# 　train_noiseにはノイズを追加
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

