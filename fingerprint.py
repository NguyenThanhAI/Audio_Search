import os.path
import uuid
import numpy as np
from pydub import AudioSegment
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter

import librosa

from utils import timethis, profile


def read_audio_file(audio_path: str, sr_desired=44100):
    y, sr = librosa.load(audio_path, sr=None)

    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired

    return y, sr


def my_spectrogram(audio, sample_rate, fft_window_size, n_mels=256):
    """Helper function that performs a spectrogram with the values in settings."""
    #nperseg = int(sample_rate * fft_window_size)
    #return spectrogram(audio, sample_rate, nperseg=nperseg)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate,
                                                     hop_length=int(sample_rate * fft_window_size),
                                                     fmin=0., fmax=sample_rate / 2.0, n_mels=n_mels)
    frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0., fmax=sample_rate / 2.0)
    timestamp = np.linspace(0, audio.shape[0] / sample_rate, num=mel_spectrogram.shape[1], endpoint=False)

    mel_spectrogram = np.log(mel_spectrogram + 1e-9)
    return frequencies, timestamp, mel_spectrogram


def file_to_spectrogram(filename, sample_rate, fft_window_size):
    """Calculates the spectrogram of a file.

    Converts a file to mono and resamples to :data:`~abracadabra.settings.SAMPLE_RATE` before
    calculating. Uses :data:`~abracadabra.settings.FFT_WINDOW_SIZE` for the window size.

    :param filename: Path to the file to spectrogram.
    :returns: * f - list of frequencies
              * t - list of times
              * Sxx - Power value for each time/frequency pair
    """
    #a = AudioSegment.from_file(filename).set_channels(1).set_frame_rate(frame_rate=sample_rate)
    #audio = np.frombuffer(a.raw_data, np.int16)
    #print(audio.shape)
    #return my_spectrogram(audio, sample_rate=sample_rate, fft_window_size=fft_window_size)
    audio, sr = read_audio_file(audio_path=filename, sr_desired=sample_rate)
    return my_spectrogram(audio=audio, sample_rate=sample_rate, fft_window_size=fft_window_size)


def find_peaks(Sxx, peak_box_size, point_efficiency):
    """Finds peaks in a spectrogram.

    Uses :data:`~abracadabra.settings.PEAK_BOX_SIZE` as the size of the region around each
    peak. Calculates number of peaks to return based on how many peaks could theoretically
    fit in the spectrogram and the :data:`~abracadabra.settings.POINT_EFFICIENCY`.

    Inspired by
    `photutils
    <https://photutils.readthedocs.io/en/stable/_modules/photutils/detection/core.html#find_peaks>`_.

    :param Sxx: The spectrogram.
    :returns: A list of peaks in the spectrogram.
    """
    data_max = maximum_filter(Sxx, size=peak_box_size, mode='constant', cval=0.0)
    peak_goodmask = (Sxx == data_max)  # good pixels are True
    y_peaks, x_peaks = peak_goodmask.nonzero()
    peak_values = Sxx[y_peaks, x_peaks]
    i = peak_values.argsort()[::-1]
    # get co-ordinates into arr
    j = [(y_peaks[idx], x_peaks[idx]) for idx in i]
    total = Sxx.shape[0] * Sxx.shape[1]
    # in a square with a perfectly spaced grid, we could fit area / PEAK_BOX_SIZE^2 points
    # use point efficiency to reduce this, since it won't be perfectly spaced
    # accuracy vs speed tradeoff
    peak_target = int((total / (peak_box_size ** 2)) * point_efficiency)
    return j[:peak_target]


def idxs_to_tf_pairs(idxs, t, f):
    """Helper function to convert time/frequency indices into values."""
    return np.array([(f[i[0]], t[i[1]]) for i in idxs])
    #return np.array(list(map(lambda x: (f[x[0]], t[x[1]]), idxs)))


def hash_point_pair(p1, p2):
    """Helper function to generate a hash from two time/frequency points."""
    return hash((p1[0], p2[0], p2[1] - p1[1]))


def target_zone(anchor, points, width, height, t):
    """Generates a target zone as described in `the Shazam paper
    <https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf>`_.

    Given an anchor point, yields all points within a box that starts `t` seconds after the point,
    and has width `width` and height `height`.

    :param anchor: The anchor point
    :param points: The list of points to search
    :param width: The width of the target zone
    :param height: The height of the target zone
    :param t: How many seconds after the anchor point the target zone should start
    :returns: Yields all points within the target zone.
    """
    x_min = anchor[1] + t
    x_max = x_min + width
    y_min = anchor[0] - (height * 0.5)
    y_max = y_min + height
    for point in points:
        if point[0] < y_min or point[0] > y_max:
            continue
        if point[1] < x_min or point[1] > x_max:
            continue
        yield point


def hash_points(points, filename, target_t, target_f, target_start):
    """Generates all hashes for a list of peaks.

    Iterates through the peaks, generating a hash for each peak within that peak's target zone.

    :param points: The list of peaks.
    :param filename: The filename of the song, used for generating song_id.
    :returns: A list of tuples of the form (hash, time offset, song_id).
    """
    hashes = []
    song_id = uuid.uuid5(uuid.NAMESPACE_OID, os.path.basename(filename)).int
    for anchor in points:
        for target in target_zone(
                anchor=anchor, points=points, width=target_t, height=target_f, t=target_start
        ):
            hashes.append((
                # hash
                hash_point_pair(p1=anchor, p2=target),
                # time offset
                anchor[1],
                # filename
                str(song_id)
            ))
    return hashes


def fingerprint_file(filename, sample_rate, fft_window_size, peak_box_size, point_efficiency, target_t, target_f,
                     target_start):
    """Generate hashes for a file.

    Given a file, runs it through the fingerprint process to produce a list of hashes from it.

    :param filename: The path to the file.
    :returns: The output of :func:`hash_points`.
    """
    f, t, Sxx = file_to_spectrogram(filename=filename, sample_rate=sample_rate, fft_window_size=fft_window_size)
    peaks = find_peaks(Sxx=Sxx, peak_box_size=peak_box_size, point_efficiency=point_efficiency)
    peaks = idxs_to_tf_pairs(idxs=peaks, t=t, f=f)
    return hash_points(points=peaks, filename=filename, target_t=target_t, target_f=target_f, target_start=target_start)


def fingerprint_audio(frames, sample_rate, fft_window_size, peak_box_size,
                      point_efficiency, target_t, target_f, target_start):
    """Generate hashes for a series of audio frames.

    Used when recording audio.

    :param frames: A mono audio stream. Data type is any that ``scipy.signal.spectrogram`` accepts.
    :returns: The output of :func:`hash_points`.
    """
    f, t, Sxx = my_spectrogram(audio=frames, sample_rate=sample_rate, fft_window_size=fft_window_size)
    peaks = find_peaks(Sxx=Sxx, peak_box_size=peak_box_size, point_efficiency=point_efficiency)
    peaks = idxs_to_tf_pairs(idxs=peaks, t=t, f=f)
    return hash_points(points=peaks, filename="recorded", target_t=target_t,
                       target_f=target_f, target_start=target_start)
