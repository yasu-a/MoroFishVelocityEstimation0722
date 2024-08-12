import time
from pathlib import Path

import cv2
from tqdm import tqdm

from config import *
from video_writer import CachedVideoFrameWriter


class VideoDump:
    def __init__(self):
        self._frames = []

    def append_frame(self, frame):
        self._frames.append(frame)

    def dump(self, path: Path, *, fps: float, start: int, end: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        with CachedVideoFrameWriter(
                str(path),
                fps=fps,
        ) as writer:
            for frame in self._frames[start:end]:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


CAPTURE_FULL_SIZE = True


def main():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_CAPTURE_FPS)

    video_base_dir = Path("./captures")

    recording = False
    record_full = False
    video_dump = None
    video_dump_full = None
    while True:
        flag, im_rgb = cap.read()
        assert flag

        xc = im_rgb.shape[1] // 2
        im_rgb_cropped = im_rgb[:, xc - VIDEO_CROP_WIDTH // 2:xc + VIDEO_CROP_WIDTH // 2, :]

        if recording:
            video_dump.append_frame(im_rgb_cropped)
            if CAPTURE_FULL_SIZE:
                video_dump_full.append_frame(im_rgb)

        im_result = im_rgb.copy()
        cv2.rectangle(
            im_result,
            (xc - VIDEO_CROP_WIDTH // 2, 0),
            (xc + VIDEO_CROP_WIDTH // 2, VIDEO_CAPTURE_HEIGHT),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            im_result,
            f"Full: {record_full}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            im_result,
            f"REC: {recording}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("win", im_result)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord("r"):
            if recording:
                recording = False
                print("Recording stopped")
                timestamp = time.time()
                video_dump.dump(
                    video_base_dir / f"{timestamp:.0f}.mp4",
                    fps=VIDEO_CAPTURE_FPS,
                    start=10,
                    end=-10,
                )
                video_dump = None
                if CAPTURE_FULL_SIZE:
                    video_dump_full.dump(
                        video_base_dir / f"{timestamp:.0f}_full.mp4",
                        fps=VIDEO_CAPTURE_FPS,
                        start=10,
                        end=-10,
                    )
                    video_dump_full = None
                print("Recording saved")
            else:
                recording = True
                video_dump = VideoDump()
                if CAPTURE_FULL_SIZE:
                    video_dump_full = VideoDump()
                print("Recording started")
        elif key == ord("s"):
            # take a screenshot
            cv2.imwrite(
                str(video_base_dir / f"{time.time():.0f}_screenshot.jpg"),
                im_rgb,
            )
            print("Screenshot saved")
        elif key == ord("f"):
            record_full = not record_full


if __name__ == '__main__':
    main()
