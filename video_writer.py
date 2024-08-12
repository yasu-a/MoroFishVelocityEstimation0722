import multiprocessing as mp
import os
import time

import imageio.v2 as iio

MAX_BACKLOG_FRAMES = 128


def _worker(q: mp.Queue, params: dict):
    out = iio.get_writer(
        params['path'],
        format='FFMPEG',
        fps=params['fps'],
        mode='I',
        codec='h264',
        quality=4,
        macro_block_size=1,
    )

    while True:
        if q.empty():
            time.sleep(0.1)
        else:
            frame = q.get(block=False)
            if frame is None:
                break
            out.append_data(frame)

    out.close()


class AsyncVideoFrameWriter:
    def __init__(self, path, fps):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.__q = mp.Queue(maxsize=MAX_BACKLOG_FRAMES)
        params = dict(path=path, fps=fps)
        self.__p = mp.Process(target=_worker, args=(self.__q, params))

    def write(self, frame):
        if not self.__p.is_alive():
            raise ValueError('writer not open')
        self.__q.put(frame)

    def setup(self):
        self.__p.start()

    def __enter__(self):
        self.setup()
        return self

    def close(self):
        self.__q.put(None)
        self.__p.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CachedVideoFrameWriter:
    def __init__(self, path, fps):
        self._path = path
        self._fps = fps
        self._frames = []

    def write(self, frame):
        self._frames.append(frame)

    def setup(self):
        pass

    def __enter__(self):
        self.setup()
        return self

    def close(self):
        with AsyncVideoFrameWriter(self._path, fps=self._fps) as writer:
            for frame in self._frames:
                writer.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
