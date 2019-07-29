from mbe.utils import AudioFile, MBEParams

import numpy as np


class AbstractPitchEstimator:

    def estimate(self, audio_file: AudioFile, params: MBEParams):
        raise NotImplemented()


class AutocorrelationPitchEstimator(AbstractPitchEstimator):

    def __init__(self, period_range: np.ndarray):
        assert len(period_range.shape) == 1
        self.period_range = period_range

    def estimate(self, audio_file: AudioFile, params: MBEParams):
        frames = audio_file.get_frames(params)
        estimates = []
        # TODO : vectorize this loop
        for frame in frames:
            window = np.hamming(params.frame_width)
            scaled_frame = np.multiply(np.multiply(window, window), frame)
            autocorr = np.correlate(scaled_frame, scaled_frame, 'full')
            autocorr_index = np.arange(-params.frame_width + 1, params.frame_width)
            rough_estimate = -1
            maximum_theta_p = -np.inf
            for period in self.period_range:
                min_k = int((-params.frame_width + 1) / period)
                max_k = int((params.frame_width - 1) / period)
                theta_p = 0
                for k in range(min_k, max_k + 1):
                    theta_p += period * autocorr[np.squeeze(np.where(autocorr_index == k * period))]
                if theta_p > maximum_theta_p:
                    maximum_theta_p = theta_p
                    rough_estimate = period
            estimates.append(rough_estimate)
        return estimates


class PySPTKPitchEstimator(AbstractPitchEstimator):

    def __init__(self, unvoicing_threshold: float):
        self.unvoicing_tresh = unvoicing_threshold

    def estimate(self, audio_file: AudioFile, params: MBEParams):
        pass

