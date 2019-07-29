from pathlib import Path
from typing import Optional

from scipy.io import wavfile
import numpy as np


class MBEParams:

    def __init__(self, frame_width: int, frame_step: int):
        self.frame_width = frame_width
        self.frame_step = frame_step


class AudioFile:

    def __init__(self, file_path: Path, start: Optional[float]=None, end:Optional[float]=None):
        """

        :param file_path:
        :param start: start time in seconds
        :param end: end time in seconds
        """
        self.rate, self.data = wavfile.read(str(file_path))
        assert len(self.data.shape) == 1

        if start is None:
            start_idx = 0.0
        else:
            start_idx = int(start * self.rate)

        if end is None:
            end_idx = self.data.shape[0]
        else:
            end_idx = int(end * self.rate)

        self.data = self.rate[start_idx:end_idx]

    def get_frames(self, params: MBEParams):
        """Returns an array of all the frames, given the MBE extraction parameters"""
        # this frame slicing method comes from
        # https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

        signal_length = self.data.shape[0]
        # Make sure that we have at least 1 frame
        num_frames = int(np.ceil(float(np.abs(signal_length - params.frame_width)) / params.frame_step))

        pad_signal_length = num_frames * params.frame_step + params.frame_width
        z = np.zeros((pad_signal_length - signal_length))
        # Pad Signal to make sure that all frames have equal number of
        # samples without truncating any samples from the original signal
        pad_signal = np.append(self.data, z)

        # creating an array of indices and using it a selector for fast slicing
        indices = np.tile(np.arange(0, params.frame_width), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * params.frame_step, params.frame_step), (params.frame_width, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        return frames
