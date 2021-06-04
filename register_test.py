from tqdm import tqdm
import librosa
from hashlib import sha1

import numpy as np

from fingerprint import fingerprint_file, file_to_spectrogram, find_peaks
from storage import DataBase
from recognize import register_directory

import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager


#audio_path = r"D:\Music\Viet\1NamNguyenTheMinh.mp3"
#hash_points = fingerprint_file(filename=audio_path, sample_rate=44100, fft_window_size=0.2, peak_box_size=20,
#                               point_efficiency=0.8, target_t=1.8, target_f=4000, target_start=0.05)
#print(len(hash_points))
#f, t, spectro = file_to_spectrogram(filename=audio_path, sample_rate=44100, fft_window_size=0.2)
#
#
#def read_audio_file(audio_path: str, sr_desired=44100):
#    y, sr = librosa.load(audio_path, sr=None)
#
#    if sr != sr_desired:
#        y = librosa.core.resample(y, sr, sr_desired)
#        sr = sr_desired
#
#    return y, sr
#
#
#print(f.shape, t.shape, spectro.shape, t)
#
#y, sr = read_audio_file(audio_path, sr_desired=44100)
#print(y.shape)
#print(y.shape[0] / sr)
#print(y.shape[0] / (0.2 * sr))
#mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=256, hop_length=int(sr * 0.2), n_fft=2048, fmin=0,
#                                                 fmax=22050)
#mel_spectrogram = np.log(mel_spectrogram + 1e-9)
#print(np.min(mel_spectrogram), np.max(mel_spectrogram))
#print(mel_spectrogram.shape)
#time_stamp = np.linspace(0, y.shape[0] / sr, num=mel_spectrogram.shape[1], endpoint=False)
#
#print(time_stamp.shape,
#      time_stamp.shape)
#
#frequencies = librosa.mel_frequencies(n_mels=256, fmin=0, fmax=22050)
#print(frequencies.shape)
#
#peaks = find_peaks(mel_spectrogram, 20, 0.9)
#print(len(peaks))
#
#hash_points = fingerprint_file(filename=audio_path, sample_rate=44100, fft_window_size=0.2, peak_box_size=20,
#                               point_efficiency=0.8, target_t=1.8, target_f=4000, target_start=0.05)
#hash_points.sort(key=lambda x: x[1])
#print(hash_points)

if __name__ == '__main__':
    BaseManager.register("DataBase", DataBase)
    manager = BaseManager()
    manager.start()
    database = manager.DataBase("D:\database_class.sqlite")
    #database = DataBase("D:\database_class.sqlite")
    database.setup_db()
    register_directory(path=r"D:\Music", num_workers=6, database=database, sample_rate=44100,
                       fft_window_size=0.2, peak_box_size=30, point_efficiency=0.8, target_t=1.8,
                       target_f=4000, target_start=0.05)

    database.close()