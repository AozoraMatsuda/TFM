import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_Green(
    kx: float, ky: float, is_edge: bool, mu: float = 0.5, E: float = 5000,
) -> np.ndarray:
    G = np.zeros([2, 2], dtype=np.complex)
    k = np.sqrt(kx * kx + ky * ky)
    if is_edge:
        G[0, 1] = 0
        G[1, 0] = 0
    else:
        gg = -mu * kx * ky
        G[0, 1] = gg
        G[1, 0] = gg
    G0 = 2 * (1 + mu) / (E * pow(k, 3))
    G[0, 0] = (1 - mu) * (k * k) + mu * (ky * ky)
    G[1, 1] = (1 - mu) * (k * k) + mu * (kx * kx)
    G *= G0
    return G


def calc_Laplacian(Tv: np.ndarray) -> np.ndarray:
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


def get_Wavefunction_in_FS(num: int, D: int) -> np.ndarray:
    # return wave function in fourier space
    ls = (
        (2 * np.pi)
        / (D * num)
        * np.hstack(
            [
                np.arange(0, num // 2 + 1, 1),
                (-1) * np.arange(np.round(num / 2 + 0.1) - 1, 0, -1),
            ]
        )
    )
    return ls


def is_edge(x: int, y: int, nCol: int, nRow: int) -> bool:
    if x == nCol // 2 + 1 or y == nRow // 2 + 1:
        return True
    return False


def convert_complex_to_vectors(df: pd.DataFrame) -> pd.DataFrame:
    R = df.apply(lambda x: x.real)
    I = df.apply(lambda x: x.imag)
    return R, I


def fft_for_vectors(df, target: str) -> np.ndarray:
    Vec = df.rearrange_by_coordinate(target)
    return np.fft.fft2(Vec)


def extract_original_values(df: pd.DataFrame) -> pd.DataFrame:
    nrow, ncol = df.shape
    return df.iloc[: nrow // 2 + 1, :]


def reconstruct_field(df: pd.DataFrame, nRow: int) -> pd.DataFrame:
    nrow, ncol = df.shape
    res = pd.DataFrame(np.zeros((nRow, ncol)))
    res.iloc[:nrow, :] = df.copy()
    for i in range(nrow):
        for j in range(ncol):
            res.iloc[-i, -j] = df.iloc[i, j].conjugate()
    return res

