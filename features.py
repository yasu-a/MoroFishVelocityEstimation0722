from abc import ABC
from typing import Sequence

import cv2
import numpy as np


class AbstractFeatureExtractor(ABC):
    def __init__(self, *, extractor, max_points):
        self._extractor = extractor
        self._max_points = max_points

    @classmethod
    def _pad_image(cls, /, image, *, size):
        pad_width = (size, size), (size, size)
        if image.ndim == 3:
            pad_width = *pad_width, (0, 0)
        return np.pad(
            image,
            pad_width=pad_width,
            mode="reflect"
        )

    @classmethod
    def _crop_key_points(
            cls,
            /,
            kp: Sequence[cv2.KeyPoint],
            *,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
    ) -> Sequence[cv2.KeyPoint]:
        return [
            cv2.KeyPoint(
                x=p.pt[0] - min_x,
                y=p.pt[1] - min_y,
                size=p.size,
                angle=p.angle,
                response=p.response,
                octave=p.octave,
                class_id=p.class_id,
            )
            for p in kp
            if min_x <= p.pt[0] <= max_x and min_y <= p.pt[1] <= max_y
        ]

    @classmethod
    def _select_keypoints(self, /, kp: Sequence[cv2.KeyPoint], *, n: int):
        res = np.array([p.response for p in kp])
        return [kp[i] for i in res.argsort()[::-1][:n]]

    @classmethod
    def _filter_edge_keypoints(
            self,
            /,
            kp: Sequence[cv2.KeyPoint],
            *,
            edge_radius,
            height,
            width,
    ):
        def is_edge(x, y):
            return (
                    x < edge_radius or
                    x >= width - edge_radius or
                    y < edge_radius or
                    y >= height - edge_radius
            )

        return [
            kp[i]
            for i in range(len(kp))
            if not is_edge(kp[i].pt[0], kp[i].pt[1])
        ]

    PAD_SIZE = 50

    def extract_features(
            self,
            /,
            im_gray: np.ndarray,
            *,
            mask: np.ndarray = None,
            remove_edges=False,
    ):
        im_padded_gray = self._pad_image(im_gray, size=self.PAD_SIZE)
        im_padded_mask = self._pad_image(mask, size=self.PAD_SIZE)
        try:
            kp = self._extractor.detect(im_padded_gray, im_padded_mask)
        except cv2.error:
            kp = []
            des = np.array([])
        else:
            kp = self._crop_key_points(
                kp,
                min_y=self.PAD_SIZE,
                max_y=self.PAD_SIZE + im_padded_gray.shape[0],
                min_x=self.PAD_SIZE,
                max_x=self.PAD_SIZE + im_padded_gray.shape[1],
            )
            if remove_edges:
                kp = self._filter_edge_keypoints(
                    kp,
                    edge_radius=3,
                    height=im_gray.shape[0],
                    width=im_gray.shape[1],
                )
            if self._max_points:
                kp = self._select_keypoints(
                    kp,
                    n=self._max_points,
                )
            kp, des = self._extractor.compute(im_padded_gray, kp)
        return kp, des


class AKAZEFeatureDescriptor(AbstractFeatureExtractor):
    def __init__(self, max_points=None):
        extractor = cv2.AKAZE.create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
            # descriptor_size=256,
            # descriptor_channels=3,
            threshold=0.002,
            nOctaves=8,
            nOctaveLayers=2,
            # diffusivity=cv2.KAZE_DIFF_PM_G2,
        )
        super().__init__(extractor=extractor, max_points=max_points)


class BRISKFeatureDescriptor(AbstractFeatureExtractor):
    # FIXME: points on left side of image not detected
    def __init__(self, max_points=None):
        extractor = cv2.BRISK.create(
            thresh=5,
            octaves=1,
            patternScale=2.0,
        )
        super().__init__(extractor=extractor, max_points=max_points)
