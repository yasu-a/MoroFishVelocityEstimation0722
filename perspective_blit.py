# https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size

import math
from typing import Tuple, Any

import cv2
import numpy as np
import numpy.typing as npt


# Input: a source image and perspective transform
# Output: a warped image and 2 translation terms
def perspective_warp(image: npt.NDArray[np.uint8], transform: npt.NDArray[Any]) -> Tuple[
    npt.NDArray[np.float32], int, int]:
    h, w = image.shape[:2]
    corners_bef = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, transform)
    xmin = math.floor(corners_aft[:, 0, 0].min())
    ymin = math.floor(corners_aft[:, 0, 1].min())
    xmax = math.ceil(corners_aft[:, 0, 0].max())
    ymax = math.ceil(corners_aft[:, 0, 1].max())
    x_adj = math.floor(xmin - corners_aft[0, 0, 0])
    y_adj = math.floor(ymin - corners_aft[0, 0, 1])
    translate = np.eye(3)
    translate[0, 2] = -xmin
    translate[1, 2] = -ymin
    corrected_transform = np.matmul(translate, transform)
    return cv2.warpPerspective(image, corrected_transform,
                               (math.ceil(xmax - xmin), math.ceil(ymax - ymin))), x_adj, y_adj


# Just like perspective_warp, but it also returns an alpha mask that can be used for blitting
def perspective_warp_with_mask(image: npt.NDArray[np.uint8], transform: npt.NDArray[Any]) -> Tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], int, int]:
    mask_in = np.empty(image.shape[:2], dtype=np.uint8)
    mask_in.fill(255)
    output, x_adj, y_adj = perspective_warp(image, transform)
    mask, _, _ = perspective_warp(mask_in, transform)
    return output, mask, x_adj, y_adj


# alpha_blits src onto dest according to the alpha values in mask at location (x, y),
# ignoring any parts that do not overlap
def alpha_blit(dest: npt.NDArray[np.uint8], src: npt.NDArray[Any], mask: npt.NDArray[Any], x: int,
               y: int) -> None:
    dl = max(x, 0)  # dest left
    dt = max(y, 0)  # dest top
    sl = max(-x, 0)  # source left
    st = max(-y, 0)  # source top
    sr = max(sl, min(src.shape[1], dest.shape[1] - x))  # source right
    sb = max(st, min(src.shape[0], dest.shape[0] - y))  # source bottom
    dr = dl + sr - sl  # dest right
    db = dt + sb - st  # dest bottom

    # Crop and convert to floats as needed
    d = dest[dt:db, dl:dr].astype(np.float32)
    s = src[st:sb, sl:sr].astype(np.float32)
    m = mask[st:sb, sl:sr].reshape((sb - st, sr - sl, 1))

    # Combine alpha channel with the mask
    if s.shape[2] > 3:
        alpha = src[st:sb, sl:sr, 3:4]
        m = np.minimum(m, alpha)
        s = s[:, :, :3]  # drop the alpha channel

    # Blit
    dest[dt:db, dl:dr] = (d * (255 - m) + s * m) / 255


# blits a perspective-warped src image onto dest
def perspective_blit(dest: npt.NDArray[np.uint8], src: npt.NDArray[np.uint8],
                     transform: npt.NDArray[Any]) -> None:
    blitme, mask, x_adj, y_adj = perspective_warp_with_mask(src, transform)
    # cv2.imwrite("blitme.png", blitme)
    alpha_blit(dest, blitme, mask, int(transform[0, 2] + x_adj), int(transform[1, 2] + y_adj))


if __name__ == '__main__':
    # Read an input image
    image: npt.NDArray[np.uint8] = cv2.imread('./captures/1721644714_screenshot.jpg')

    # Make a perspective transform
    h, w = image.shape[:2]
    corners_in = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
    corners_out = np.array([[[100, 100]], [[300, -100]], [[500, 300]], [[-50, 500]]],
                           dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners_in, corners_out)

    # Blit the warped image on top of the original
    perspective_blit(image, image, transform)
    cv2.imwrite('output.jpg', image)
