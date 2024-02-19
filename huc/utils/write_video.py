import cv2
import numpy as np


def write_video(filepath, fps, rgb_images):
    """
    Writes a video from images.
    Args:
      filepath: Path where the video will be saved.
      fps: the video writing fps.
      rgb_images: the rgb images that will be drawn into the video.
      width: the video viewport width.
      height: the video viewport height.
    Raises:
      ValueError: If frames per second (fps) is not set (set_fps is not called)
    """
    input_examine_array = np.array(rgb_images[0].copy(), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if len(input_examine_array.shape) == 4:
        out = cv2.VideoWriter(filepath, fourcc, fps, tuple([rgb_images[0][0].shape[1], rgb_images[0][0].shape[0]]))
        for batch in rgb_images:
            for img in batch:
                out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif len(input_examine_array.shape) == 3 or len(input_examine_array.shape) == 2:
        out = cv2.VideoWriter(filepath, fourcc, fps, tuple([rgb_images[0].shape[1], rgb_images[0].shape[0]]))
        for img in rgb_images:
            out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f'Input array must be 4 or less than 4 dimensional. Now only has {len(input_examine_array.shape)} dimensions.')
    out.release()
    print(f'The video has been made and released to: {filepath}.\n')
