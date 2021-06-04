import os
import logging
from multiprocessing import Pool, Lock, current_process, Manager
from functools import partial
import numpy as np
from tinytag import TinyTag
from record import record_audio
from fingerprint import fingerprint_file, fingerprint_audio
from storage import DataBase

KNOWN_EXTENSIONS = ["mp3", "wav", "flac", "m4a"]


def get_song_info(filename):
    tag = TinyTag.get(filename)
    artist = tag.artist if tag.albumartist is None else tag.albumartist
    return (artist, tag.album, tag.title)


def register_song(filename, database: DataBase, sample_rate, fft_window_size, peak_box_size,
                  point_efficiency, target_t, target_f, target_start, lock):
    if database.song_in_db(filename=filename):
        print("Song: {} already in database".format(filename))
        return
    hashes = fingerprint_file(filename, sample_rate=sample_rate, fft_window_size=fft_window_size,
                              peak_box_size=peak_box_size,
                              point_efficiency=point_efficiency, target_t=target_t,
                              target_f=target_f, target_start=target_start)
    song_info = get_song_info(filename)
    try:
        logging.info(f"{current_process().name} waiting to write {filename}")
        lock.acquire()
        logging.info(f"{current_process().name} writing {filename}")
        database.store_song(hashes, song_info)
        logging.info(f"{current_process().name} wrote {filename}")
        lock.release()
    except NameError:
        logging.info(f"Single-threaded write of {filename}")
        # running single-threaded, no lock needed
        database.store_song(hashes, song_info)
    #store_song(hashes=hashes, song_info=song_info, db_path=db_path)


def register_directory(path, num_workers, database: DataBase, sample_rate, fft_window_size, peak_box_size,
                       point_efficiency, target_t, target_f, target_start):
    #def pool_init(l):
    #    global lock
    #    lock = l
    #    logging.info(f"Pool init in {current_process().name}")

    to_register = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] not in KNOWN_EXTENSIONS:
                continue
            file_path = os.path.join(path, root, f)
            to_register.append(file_path)
    #l = Lock()
    m = Manager()
    l = m.Lock()
    register_a_song = partial(register_song, database=database, sample_rate=sample_rate, fft_window_size=fft_window_size,
                              peak_box_size=peak_box_size,
                              point_efficiency=point_efficiency, target_t=target_t, target_f=target_f,
                              target_start=target_start, lock=l)
    #with Pool(processes=num_workers, initializer=pool_init, initargs=(l,)) as p:
    #    p.map(register_a_song, to_register)
    print("Number of song: {}".format(len(to_register)))
    with Pool(processes=num_workers) as p:
        p.map(register_a_song, to_register)
    # speed up future reads
    database.checkpoint_db()


def score_match(offsets):
    # Use bins spaced 0.5 seconds apart
    binwidth = 0.5
    tks = list(map(lambda x: x[0] - x[1], offsets))
    hist, _ = np.histogram(tks,
                           bins=np.arange(int(min(tks)),
                                          int(max(tks)) + binwidth + 1,
                                          binwidth))
    return np.max(hist)


def best_match(matches):
    matched_song = None
    best_score = 0
    for song_id, offsets in matches.items():
        if len(offsets) < best_score:
            # can't be best score, avoid expensive histogram
            continue
        score = score_match(offsets)
        if score > best_score:
            best_score = score
            matched_song = song_id
    return matched_song


def recognize_song(filename, sample_rate, fft_window_size, peak_box_size,
                   point_efficiency, target_t, target_f, target_start, database: DataBase, threshold):
    hashes = fingerprint_file(filename=filename, sample_rate=sample_rate,
                              fft_window_size=fft_window_size, peak_box_size=peak_box_size,
                              point_efficiency=point_efficiency, target_t=target_t,
                              target_f=target_f, target_start=target_start)
    matches = database.get_matches(hashes=hashes, threshold=threshold)
    matched_song = best_match(matches=matches)
    info = database.get_info_for_song_id(song_id=matched_song)
    if info is not None:
        return info
    return matched_song


def listen_to_song(filename, format, channels, rate, chunk, record_seconds,
                   sample_rate, fft_window_size, peak_box_size, point_efficiency, target_t, target_f, target_start,
                   database: DataBase, threshold=5):
    audio = record_audio(filename=filename, format=format, channels=channels,
                         rate=rate, chunk=chunk, record_seconds=record_seconds)
    hashes = fingerprint_audio(frames=audio, sample_rate=sample_rate, fft_window_size=fft_window_size,
                               peak_box_size=peak_box_size, point_efficiency=point_efficiency,
                               target_t=target_t, target_f=target_f, target_start=target_start)
    matches = database.get_matches(hashes=hashes)
    matched_song = best_match(matches=matches)
    info = database.get_info_for_song_id(matched_song)
    if info is not None:
        return info
    return matched_song
