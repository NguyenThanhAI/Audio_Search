import os
import argparse
import time
import sqlite3
import uuid

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from fingerprint import fingerprint_file, read_audio_file, fingerprint_audio
from recognize import recognize_song, get_matches, best_match, get_info_for_song_id


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
    parser.add_argument("--query_song", type=str, default=None)
    parser.add_argument("--song_mode", type=str2bool, default=True)
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
    query_song = args.query_song
    song_mode = args.song_mode
    sample_rate = args.sample_rate
    fft_window_size = args.fft_window_size
    peak_box_size = args.peak_box_size
    point_efficiency = args.point_efficiency
    target_t = args.target_t
    target_f = args.target_f
    target_start = args.target_start

    assert 0. < fft_window_size < 1.
    assert 0. < point_efficiency <= 1.

    if song_mode:
        song = recognize_song(filename=query_song, db_path=db_path, sample_rate=sample_rate, fft_window_size=fft_window_size,
                              peak_box_size=peak_box_size, point_efficiency=point_efficiency, target_t=target_t,
                              target_f=target_f, target_start=target_start, threshold=5)
        print("Song mode, recognized song: {}".format(song))

    else:
        audio, sr = read_audio_file(audio_path=query_song, sr_desired=sample_rate)
        hashes = fingerprint_audio(frames=audio, sample_rate=sample_rate, fft_window_size=fft_window_size,
                                   peak_box_size=peak_box_size, point_efficiency=point_efficiency, target_t=target_t,
                                   target_f=target_f, target_start=target_start)
        matches = get_matches(hashes=hashes, db_path=db_path, threshold=5)
        matched_song = best_match(matches=matches)
        song = get_info_for_song_id(song_id=matched_song, db_path=db_path)

        print("Audio mode, recognized song: {}".format(song))
