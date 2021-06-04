import os
import argparse
import time
import sqlite3
import uuid

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import pyaudio

from fingerprint import fingerprint_file, read_audio_file, fingerprint_audio
from recognize import recognize_song, best_match, listen_to_song
from storage import DataBase


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--record_seconds", type=int, default=10)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--fft_window_size", type=float, default=0.2)
    parser.add_argument("--peak_box_size", type=int, default=30)
    parser.add_argument("--point_efficiency", type=float, default=0.8)
    parser.add_argument("--target_t", type=float, default=1.8)
    parser.add_argument("--target_f", type=int, default=4000)
    parser.add_argument("--target_start", type=float, default=0.05)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    db_path = args.db_path
    channels = args.channels
    chunk = args.chunk
    record_seconds = args.record_seconds
    sample_rate = args.sample_rate
    fft_window_size = args.fft_window_size
    peak_box_size = args.peak_box_size
    point_efficiency = args.point_efficiency
    target_t = args.target_t
    target_f = args.target_f
    target_start = args.target_start

    database = DataBase(db_path=db_path)

    song = listen_to_song(filename=None, format=pyaudio.paInt16, channels=channels, rate=sample_rate, chunk=chunk,
                          record_seconds=record_seconds,
                          sample_rate=sample_rate, fft_window_size=fft_window_size, peak_box_size=peak_box_size,
                          point_efficiency=point_efficiency,
                          target_t=target_t, target_f=target_f, target_start=target_start, database=database)

    print("Recognized song: {}".format(song))
