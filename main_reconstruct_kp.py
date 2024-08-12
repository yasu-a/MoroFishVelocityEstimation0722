import cv2
import numpy as np
import pandas as pd

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
    im_mask = cv2.dilate(im_mask, _morph_filter, iterations=3)
    im_mask = cv2.erode(im_mask, _morph_filter, iterations=3)

    return im_mask


feature_extractor = AKAZEFeatureDescriptor()

PREVIEW_BG_MASKS = False
PREVIEW_KEYPOINTS = False
PREVIEW_MATCHES = False


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

    # estimate foreground
    masks_fg = np.zeros(
        shape=frames.shape[:-1],
        dtype=np.uint8,
    )
    for i in range(len(frames)):
        masks_fg[i] = subtract_background(frames[i])

        if PREVIEW_BG_MASKS:
            im_result = frames[i].copy()
            im_result[masks_fg[i] == 0] //= 6
            cv2.imshow("bg_mask", im_result)
            cv2.waitKey(1)
    if PREVIEW_BG_MASKS:
        cv2.destroyWindow("bg_mask")

    # extract features
    frame_kps, frame_des = [], []
    for i in range(len(frames)):
        frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        kp, des = feature_extractor.extract_features(frame_gray, mask=masks_fg[i])
        frame_kps.append(kp)
        frame_des.append(des)

        if PREVIEW_KEYPOINTS:
            im_result = cv2.drawKeypoints(
                frames[i],
                frame_kps[i],
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            cv2.imshow("keypoints", im_result)
            cv2.waitKey(1)
    if PREVIEW_KEYPOINTS:
        cv2.destroyWindow("keypoints")

    # match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    frame_matches = []
    for i in range(len(frames) - 1):
        if frame_des[i + 1] is None or frame_des[i] is None:
            matches = None
        else:
            matches = matcher.match(queryDescriptors=frame_des[i + 1],
                                    trainDescriptors=frame_des[i])

        frame_matches.append(matches)

        if PREVIEW_MATCHES:
            im_result = cv2.drawMatches(
                frames[i],
                frame_kps[i],
                frames[i + 1],
                frame_kps[i + 1],
                matches,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow("matches", im_result)
            cv2.waitKey(1)
    if PREVIEW_MATCHES:
        cv2.destroyWindow("matches")

    # find transform
    frame_mats = []
    for i in range(len(frames) - 1):
        mat = None
        if frame_matches[i] is not None:
            point_start = np.array([
                [
                    frame_kps[i][m.trainIdx].pt[0],
                    frame_kps[i][m.trainIdx].pt[1],
                ] for m in frame_matches[i]
            ])
            point_end = np.array([
                [
                    frame_kps[i + 1][m.queryIdx].pt[0],
                    frame_kps[i + 1][m.queryIdx].pt[1],
                ] for m in frame_matches[i]
            ])
            if len(point_start) >= 4:
                mat, _ = cv2.findHomography(
                    point_start,
                    point_end,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1,
                )
        frame_mats.append(mat)

    for i in range(len(frames) - 1):
        if frame_mats[i] is not None:
            h, w = frames[i].shape[:2]
            im_result = cv2.warpPerspective(
                frames[i + 1],
                np.linalg.inv(frame_mats[i]

                              ) + [[0, 0, w], [0, 0, h], [0, 0, 0]],
                (w * 3, h * 3),
            )
            im_result[h:h * 2, w:w * 2, :] \
                = cv2.addWeighted(im_result[h:h * 2, w:w * 2, :], 0.5, frames[i], 0.5, 0)
            cv2.imshow("win", cv2.resize(im_result, None, fx=1 / 3, fy=1 / 3))
            cv2.waitKey()
            print(pd.DataFrame(frame_mats[i]).round(2))


if __name__ == '__main__':
    main()
