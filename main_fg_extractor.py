import os

import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from abc import ABC
from typing import Sequence, Iterable
import h5py

_background_subtractor = cv2.createBackgroundSubtractorMOG2(
    detectShadows=True,
    history=256,
    varThreshold=16,
)
# _background_subtractor = cv2.createBackgroundSubtractorKNN(
#     history=256,
#     dist2Threshold=6,
#     detectShadows=True,
# )
# _background_subtractor = cv2.createBackgroundSubtractorGMG(
#
# )
_morph_filter = np.ones((3, 3))


def subtract_background(im):
    im_mask = _background_subtractor.apply(im)
    _, im_mask = cv2.threshold(im_mask, 254, 255, cv2.THRESH_BINARY)
    im_mask = cv2.dilate(im_mask, _morph_filter, iterations=4)
    im_mask = cv2.erode(im_mask, _morph_filter, iterations=3)

    return im_mask


def calc_flow(im_prev, im_curr):
    flow = cv2.calcOpticalFlowFarneback(
        im_prev,
        im_curr,
        flow=None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow


class VideoData:
    def __init__(self, video_path: str, gc_sample_regions: tuple[slice, slice], scaling=1.0):
        self._video_path = video_path
        self._gc_sample_regions = gc_sample_regions
        self._scaling = scaling

        self._n: int | None = None
        self._w: int | None = None
        self._h: int | None = None
        self._frame_lst: np.ndarray | None = None
        self._frame_gc_lst: np.ndarray | None = None  # auto gamma correction
        self._gray_lst: np.ndarray | None = None
        self._gray_gc_lst: np.ndarray | None = None  # auto gamma correction
        self._pca_gray_lst: np.ndarray | None = None
        self._mask_lst: np.ndarray | None = None
        self._flow_lst: np.ndarray | None = None

    _OBJECT_NAMES = [
        "frame_lst",
        "frame_gc_lst",
        "gray_lst",
        "gray_gc_lst",
        "pca_gray_lst",
        "mask_lst",
        "flow_lst",
    ]

    @property
    def n(self):
        return self._n

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def frame_lst(self):
        return self._frame_lst

    @property
    def frame_gc_lst(self):
        return self._frame_gc_lst

    @property
    def gray_lst(self):
        return self._gray_lst

    @property
    def gray_gc_lst(self):
        return self._gray_gc_lst

    @property
    def pca_gray_lst(self):
        return self._pca_gray_lst

    @property
    def mask_lst(self):
        return self._mask_lst

    @property
    def flow_lst(self):
        return self._flow_lst

    @property
    def flow_x_lst(self):
        return self._flow_lst[:, :, :, 0]

    @property
    def flow_y_lst(self):
        return self._flow_lst[:, :, :, 1]

    def _create_frame_lst(self):
        cap = cv2.VideoCapture(self._video_path)
        assert cap.isOpened()

        self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self._scaling)
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self._scaling)

        self._frame_lst = np.empty(
            shape=(self._n, self._h, self._w, 3),
            dtype=np.uint8,
        )
        i = 0
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            self._frame_lst[i, :, :, :] = cv2.resize(frame, (self._w, self._h))
            i += 1
        # self._frame_lst = self._frame_lst[::-1]
        self._frame_lst.setflags(write=False)

    def _create_frame_gc_lst(self):
        self._frame_gc_lst = np.empty(
            shape=(self._n, self._h, self._w, 3),
            dtype=np.uint8,
        )
        for i in tqdm(range(self._n)):
            mean = np.mean(
                self._frame_lst[i, :, :, :][self._gc_sample_regions],
            )
            gamma = np.log(127) / np.log(mean)
            lut = np.power(np.arange(256, dtype=np.float32), gamma).clip(0, 255).astype(np.uint8)
            cv2.LUT(self._frame_lst[i], lut, self._frame_gc_lst[i])
        self._frame_gc_lst.setflags(write=False)

    def _create_gray_lst(self):
        self._gray_lst = np.empty(
            shape=(self._n, self._h, self._w),
            dtype=np.uint8,
        )
        for i in tqdm(range(self._n)):
            self._gray_lst[i] = cv2.cvtColor(self._frame_lst[i], cv2.COLOR_BGR2GRAY)
        self._gray_lst.setflags(write=False)

    def _create_gray_gc_lst(self):
        self._gray_gc_lst = np.empty(
            shape=(self._n, self._h, self._w),
            dtype=np.uint8,
        )
        for i in range(self._n):
            mean = np.mean(
                self._gray_lst[i, :, :][self._gc_sample_regions],
            )
            gamma = np.log(127) / np.log(mean)
            lut = np.power(np.arange(256, dtype=np.float32), gamma).clip(0, 255).astype(np.uint8)
            cv2.LUT(self._gray_lst[i], lut, self._gray_gc_lst[i])
        self._gray_gc_lst.setflags(write=False)

    def _create_pca_gray_lst(self):
        self._pca_gray_lst = np.empty(
            shape=(self._n, self._h, self._w),
            dtype=np.uint8,
        )
        pca = PCA(n_components=1)
        n_samples = 2 ** 17
        idx_frame = np.arange(self._n, dtype=np.uint16)
        idx_height = np.arange(self._h, dtype=np.uint16)
        idx_width = np.arange(self._w, dtype=np.uint16)
        points = (
            np.random.choice(idx_frame, size=n_samples),
            np.random.choice(idx_height, size=n_samples),
            np.random.choice(idx_width, size=n_samples),
        )
        points_pca = pca.fit_transform(self._frame_gc_lst[points])
        v_min, v_max = points_pca.min(), points_pca.max()
        for i in tqdm(range(self._n)):
            a = pca.transform(self._frame_lst[i].reshape(-1, 3)).reshape(self._h, self._w)
            a = np.uint8(np.clip((a - v_min) / (v_max - v_min) * 255, 0, 255))
            self._pca_gray_lst[i, :, :] = a
        self._pca_gray_lst.setflags(write=False)

    def _create_mask_lst(self):
        self._mask_lst = np.empty(
            shape=(self._n, self._h, self._w),
            dtype=np.uint8,
        )
        for i in tqdm(range(self._n)):
            self._mask_lst[i] = subtract_background(self._frame_lst[i])
        self._mask_lst.setflags(write=False)

    def _create_flow_lst(self):
        self._flow_lst = np.empty(
            shape=(self._n, self._h, self._w, 2),
            dtype=np.float32,
        )
        for i in tqdm(range(self._n - 1)):
            if i == 0:
                self._flow_lst[i, :, :, :] = 0
            else:
                self._flow_lst[i] = calc_flow(
                    im_prev=self._gray_lst[i],
                    im_curr=self._gray_lst[i + 1],
                )
        self._flow_lst.setflags(write=False)

    @property
    def h5_path(self) -> str:
        dir_name, file_name = os.path.split(self._video_path)
        file_name_body, _ = os.path.splitext(file_name)
        return os.path.join(dir_name, f"{file_name_body}.h5")

    def save(self):
        path = self.h5_path
        with h5py.File(path, "w") as f:
            for name in self._OBJECT_NAMES:
                f.create_dataset(name, data=getattr(self, f"_{name}"))

    def load(self):
        path = self.h5_path
        with h5py.File(path, "r") as f:
            for name in self._OBJECT_NAMES:
                setattr(self, f"_{name}", f[name][...])
                getattr(self, f"_{name}").setflags(write=False)

            self._n, self._h, self._w = self._frame_lst.shape[:3]

    def create_all(self):
        try:
            self.load()
        except FileNotFoundError:
            for name in self._OBJECT_NAMES:
                getattr(self, f"_create_{name}")()
            self.save()

    def create_debug_image(
            self, i: int,
            sources: list[str | tuple[str, int] | np.ndarray],
            value_scale: float = 5.0,  # scaling factor for normalizing float image
            scaling: float = 1.0,
    ) -> np.ndarray:
        """
        :param i:
             - frame index
        :param sources:
             - str: source name
             - tuple[str, int]: source name and frame index
             - np.ndarray: frame array
        :return:
        """
        frame_lst = []
        for source_item in sources:
            frame_index = i
            if isinstance(source_item, tuple):
                source_item, frame_index = source_item
            if isinstance(source_item, str):
                frame = getattr(self, source_item)[frame_index, ...]
            else:
                frame = source_item
            assert isinstance(frame, np.ndarray), (i, sources, type(frame))
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            assert frame.ndim == 3, (i, sources, frame.ndim)
            if frame.dtype != np.uint8:
                frame = np.uint8(
                    (np.clip(frame, -value_scale, +value_scale) / value_scale + 1) / 2 * 255,
                )
            frame_lst.append(frame)

        im_debug = np.hstack(frame_lst)
        im_debug = cv2.resize(im_debug, None, fx=scaling, fy=scaling)
        return im_debug

    def __len__(self):
        return self._n

    def __iter__(self) -> Iterable[int]:
        yield from range(self._n)


class ColorFGEstimator:
    def __init__(self, data: VideoData, bgr_source: str = "frame_lst"):
        self._data = data
        self._bgr_source = bgr_source

        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None

        self._mask_lst = np.empty_like(data.gray_lst)

    @property
    def mask_lst(self):
        return self._mask_lst

    @property
    def _bgr_lst(self):
        return getattr(self._data, self._bgr_source)

    def train(self, i_end: int):
        self._mean = self._bgr_lst[:i_end].mean(axis=0, dtype=np.float32)
        diff = self._bgr_lst - self._mean
        self._cov_inv = np.linalg.pinv(
            np.mean(
                np.einsum(
                    "ijka,ijkb->ijkab",
                    diff,
                    diff,
                    dtype=np.float32,
                ),
                axis=0,
            )
        )
        self._cov_inv += np.eye(self._cov_inv.shape[-1])[None, None, :, :] * 5

    def create_mask(self, sigma: float):
        for i in tqdm(self._data):
            a = self._bgr_lst[i]
            diff = a - self._mean
            mahal_sq = np.einsum(
                "ija,ija->ij",
                diff,
                np.einsum(
                    "ijab,ijb->ija",
                    self._cov_inv,
                    diff,
                    dtype=np.float32,
                )
            )
            mahal_sq = np.sqrt(mahal_sq)

            self._mask_lst[i] = np.uint8(mahal_sq > sigma) * 255
            # cv2.morphologyEx(
            #     self._mask_lst[i],
            #     cv2.MORPH_OPEN,
            #     np.ones((3, 3)),
            #     self._mask_lst[i],
            #     iterations=3,
            # )
        self._mask_lst.setflags(write=False)


class ColorNormFGEstimator:
    def __init__(self, data: VideoData, bgr_source: str = "frame_lst"):
        self._data = data
        self._bgr_source = bgr_source

        self._mean: np.ndarray | None = None

        self._mask_lst = np.empty_like(data.gray_lst)

    @property
    def mask_lst(self):
        return self._mask_lst

    @property
    def _bgr_lst(self):
        return getattr(self._data, self._bgr_source)

    def train(self, i_end: int):
        self._mean = self._bgr_lst[:i_end].mean(axis=0, dtype=np.float32)

    def create_mask(self, dist_thresh: float):
        for i in tqdm(self._data):
            a = self._bgr_lst[i]
            diff = np.linalg.norm(a - self._mean, axis=-1)

            self._mask_lst[i] = np.uint8(diff >= dist_thresh) * 255
            cv2.morphologyEx(
                self._mask_lst[i],
                cv2.MORPH_OPEN,
                np.ones((3, 3)),
                self._mask_lst[i],
                iterations=3,
            )
        self._mask_lst.setflags(write=False)


class GrayFGEstimator:
    def __init__(
            self,
            data: VideoData,
            gray_source: str = "gray_gc_lst",
            n_iter_morph_open: int = 3,
    ):
        self._data = data
        self._gray_source = gray_source
        self._n_iter_morph_open = n_iter_morph_open

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

        self._mask_lst = np.empty_like(data.gray_lst)

    @property
    def mask_lst(self):
        return self._mask_lst

    @property
    def _gray_lst(self):
        return getattr(self._data, self._gray_source)

    def train(self, i_end: int, std_offset: float):
        self._mean = self._gray_lst[:i_end].mean(axis=0)
        self._std = self._gray_lst[:i_end].std(axis=0)
        # self._std += std_offset
        self._std = np.maximum(self._std, std_offset)

    def create_mask(self, sigma: float):
        v_min = self._mean - self._std * sigma
        v_max = self._mean + self._std * sigma
        for i in self._data:
            self._mask_lst[i] = ~((v_min <= self._gray_lst[i]) & (self._gray_lst[i] <= v_max))
            self._mask_lst[i] *= 255
            cv2.morphologyEx(
                self._mask_lst[i],
                cv2.MORPH_OPEN,
                np.ones((3, 3)),
                self._mask_lst[i],
                iterations=self._n_iter_morph_open,
            )
        self._mask_lst.setflags(write=False)


# def main():
#     data = VideoData("./captures/3.mp4", auto_gamma_sample_y_end_ratio=0.5, scaling=1)
#     data.create_all()
#     fge_no_gc = ColorFGEstimator(data, bgr_source="frame_lst")
#     fge_no_gc.train(i_end=50)
#     fge_no_gc.create_mask(sigma=4)
#     fge_with_gc = ColorFGEstimator(data, bgr_source="frame_gc_lst")
#     fge_with_gc.train(i_end=50)
#     fge_with_gc.create_mask(sigma=4)
#
#     for i in data:
#         im_debug = data.create_debug_image(
#             i,
#             sources=[
#                 "frame_lst",
#                 "frame_gc_lst",
#                 "pca_gray_lst",
#                 # "flow_x_lst",
#                 # "flow_y_lst",
#                 fge_no_gc.mask_lst[i],
#                 fge_with_gc.mask_lst[i],
#             ],
#         )
#         cv2.imshow("win", im_debug)
#         cv2.setWindowTitle("win", f"#{i}")
#         cv2.waitKey()


def main():
    data = VideoData(
        "./captures/3.mp4",
        gc_sample_regions=(slice(0, 100), slice(None, None)),
        scaling=1,
    )
    data.create_all()

    sigma = 4
    i_end = 48
    n_iter_morph_open = 3
    std_offset = 4

    fge_gray = GrayFGEstimator(
        data,
        gray_source="gray_lst",
        n_iter_morph_open=n_iter_morph_open,
    )
    fge_gray.train(i_end=i_end, std_offset=std_offset)
    fge_gray.create_mask(sigma=sigma)

    fge_gray_gc = GrayFGEstimator(
        data,
        gray_source="gray_gc_lst",
        n_iter_morph_open=n_iter_morph_open,
    )
    fge_gray_gc.train(i_end=i_end, std_offset=std_offset)
    fge_gray_gc.create_mask(sigma=sigma)

    fge_col_norm = ColorNormFGEstimator(
        data,
        bgr_source="frame_lst",
    )
    fge_col_norm.train(i_end=i_end)
    fge_col_norm.create_mask(dist_thresh=70)

    i = 0
    while True:
        im_debug = data.create_debug_image(
            i,
            sources=[
                "frame_lst",
                "gray_lst",
                "gray_gc_lst",
                # "flow_x_lst",
                # "flow_y_lst",
                fge_gray.mask_lst[i],
                fge_gray_gc.mask_lst[i],
                fge_col_norm.mask_lst[i],
            ],
            scaling=1,
        )
        cv2.imshow("win", im_debug)
        cv2.setWindowTitle("win", f"#{i}")
        k = cv2.waitKey()
        if k == ord("a"):
            i -= 1
        elif k == ord("d"):
            i += 1
        elif k == ord("z"):
            i -= 10
        elif k == ord("c"):
            i += 10
        elif k == ord("q"):
            break
        i = max(0, min(len(data) - 1, i))


if __name__ == '__main__':
    main()
