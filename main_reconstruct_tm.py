import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from features import AKAZEFeatureDescriptor

_background_subtractor = cv2.createBackgroundSubtractorMOG2(
    detectShadows=True,
    history=512,
    varThreshold=256,
)
_morph_filter = np.ones((3, 3))


def subtract_background(im):
    im_mask = _background_subtractor.apply(im)
    _, im_mask = cv2.threshold(im_mask, 254, 255, cv2.THRESH_BINARY)
    im_mask = cv2.dilate(im_mask, _morph_filter, iterations=5)
    im_mask = cv2.erode(im_mask, _morph_filter, iterations=3)

    return im_mask


feature_extractor = AKAZEFeatureDescriptor()

PREVIEW_BG_MASKS = False
PREVIEW_KEYPOINTS = False
PREVIEW_MATCHES = False


def imshow(img):
    plt.imshow(img)
    plt.show()
    plt.close()


def calculate_movement(
        im_prev,
        im_cur,
        mask_prev,
        cancel_sq_thresh=1.0,
        tm_margin_x=8,
        tm_margin_y=8,
        # accept_sq_thresh=0.1,
) -> tuple[float, float] | None:
    # skip frames with no difference
    sq_diff = np.mean(np.square(im_prev - im_cur))
    if sq_diff < cancel_sq_thresh:
        return None

    # template matching
    frame_cur_pad = np.pad(
        im_cur,
        pad_width=(
            (tm_margin_y, tm_margin_y),
            (tm_margin_x, tm_margin_x),
            (0, 0),
        ),
    )
    im_match = cv2.matchTemplate(
        image=frame_cur_pad,
        templ=im_prev,
        method=cv2.TM_SQDIFF_NORMED,
        mask=mask_prev,
    )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(im_match)
    # if min_val > accept_sq_thresh:
    #     return None
    match_ofs_top_left = np.array(min_loc) - [tm_margin_x, tm_margin_y]
    return match_ofs_top_left - [0, 0]


def calculate_transformation(
        im_prev,
        im_cur,
        mask_prev,
) -> np.ndarray:
    ofs = calculate_movement(
        im_prev,
        im_cur,
        mask_prev,
    )
    if ofs is None:
        return np.eye(3)
    else:
        mx, my = ofs
        return np.array([
            [1, 0, mx],
            [0, 1, my],
            [0, 0, 1],
        ])


def main():
    cap = cv2.VideoCapture("./captures/1721644730.mp4")
    assert cap.isOpened()

    # extract frames
    frames = np.zeros(
        shape=(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3,
        ),
        dtype=np.uint8,
    )
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frames[int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, :, :, :] = frame
    frames = frames[30:-60]

    h, w = frames[0].shape[:2]
    print(f"{w=}, {h=}")

    # estimate foreground
    masks_fg = np.zeros(
        shape=frames.shape[:-1],
        dtype=np.uint8,
    )
    for i in tqdm(range(len(frames)), desc="Estimate foreground"):
        masks_fg[i] = subtract_background(frames[i])

        if PREVIEW_BG_MASKS:
            im_result = frames[i].copy()
            im_result[masks_fg[i] == 0] //= 6
            cv2.imshow("bg_mask", im_result)
            cv2.waitKey(1)
    if PREVIEW_BG_MASKS:
        cv2.destroyWindow("bg_mask")

    frames_gasus = []
    for f in frames:
        frames_gasus.append(
            cv2.GaussianBlur(f, (7, 7), 0),
        )

    # find match transformations
    frame_mat = []
    for i in tqdm(range(len(frames)), desc="Find match positions"):
        mat = calculate_transformation(
            im_prev=frames_gasus[i - 1],
            im_cur=frames_gasus[i],
            mask_prev=masks_fg[i - 1],
        )
        frame_mat.append(mat)

    # calculate offsets
    frame_mats_int = []
    for i in range(len(frame_mat)):
        mat = frame_mat[i]
        if not frame_mats_int:
            frame_mats_int.append(mat)
        else:
            frame_mats_int.append(
                frame_mats_int[-1] @ mat,
            )
    # stitch images
    margin = np.array([100, 100])
    size = np.array([w * 20, h]) + margin * 2
    ofs = np.array([w * 10, 0]) + margin
    im_stitch = np.zeros(
        shape=(*size[::-1], 3),
        dtype=np.uint8,
    )
    for i in tqdm(range(len(frames)), desc="Stitching images"):
        mat = np.linalg.inv(frame_mats_int[i])
        mat += [[0, 0, ofs[0]], [0, 0, ofs[1]], [0, 0, 0]]
        frame_cur, frame_prev = frames[i], frames[i - 1]
        im_result = cv2.warpPerspective(
            frame_prev,
            mat,
            im_stitch.shape[:2][::-1],
        )
        im_mask = np.ones(
            shape=(h, w),
            dtype=np.float32,
        )
        im_mask = cv2.warpPerspective(
            im_mask,
            mat,
            im_stitch.shape[:2][::-1],
        )
        im_mask = np.bool_(im_mask)
        im_stitch[im_mask] = im_result[im_mask]
        cv2.imshow("win", im_stitch)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
