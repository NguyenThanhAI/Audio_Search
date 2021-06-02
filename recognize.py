import os
import logging
from multiprocessing import Pool, Lock, current_process, Manager
from functools import partial
import numpy as np
from tinytag import TinyTag
from record import record_audio
from fingerprint import fingerprint_file, fingerprint_audio
from storage import store_song, get_matches, get_info_for_song_id, song_in_db, checkpoint_db

KNOWN_EXTENSIONS = ["mp3", "wav", "flac", "m4a"]


def get_song_info(filename):
    """Gets the ID3 tags for a file. Returns None for tuple values that don't exist.

    :param filename: Path to the file with tags to read
    :returns: (artist, album, title)
    :rtype: tuple(str/None, str/None, str/None)
    """
    tag = TinyTag.get(filename)
    artist = tag.artist if tag.albumartist is None else tag.albumartist
    return (artist, tag.album, tag.title)


def register_song(filename, db_path, sample_rate, fft_window_size, peak_box_size,
                  point_efficiency, target_t, target_f, target_start, lock):
    """Register a single song.

    Checks if the song is already registered based on path provided and ignores
    those that are already registered.

    :param filename: Path to the file to register"""
    if song_in_db(filename, db_path=db_path):
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
        store_song(hashes, song_info, db_path=db_path)
        logging.info(f"{current_process().name} wrote {filename}")
        lock.release()
    except NameError:
        logging.info(f"Single-threaded write of {filename}")
        # running single-threaded, no lock needed
        store_song(hashes, song_info, db_path=db_path)
    #store_song(hashes=hashes, song_info=song_info, db_path=db_path)


def register_directory(path, num_workers, db_path, sample_rate, fft_window_size, peak_box_size,
                       point_efficiency, target_t, target_f, target_start):
    """Recursively register songs in a directory.

    Uses :data:`~abracadabra.settings.NUM_WORKERS` workers in a pool to register songs in a
    directory.

    :param path: Path of directory to register
    """
    #def pool_init(l):
    #    """Init function that makes a lock available to each of the workers in
    #    the pool. Allows synchronisation of db writes since SQLite only supports
    #    one writer at a time.
    #    """
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
    register_a_song = partial(register_song, db_path=db_path, sample_rate=sample_rate, fft_window_size=fft_window_size,
                              peak_box_size=peak_box_size,
                              point_efficiency=point_efficiency, target_t=target_t, target_f=target_f,
                              target_start=target_start, lock=l)
    #with Pool(processes=num_workers, initializer=pool_init, initargs=(l,)) as p:
    #    p.map(register_a_song, to_register)
    print("Number of song: {}".format(len(to_register)))
    with Pool(processes=num_workers) as p:
        p.map(register_a_song, to_register)
    # speed up future reads
    checkpoint_db(db_path=db_path)


def score_match(offsets):
    """Score a matched song.

    Calculates a histogram of the deltas between the time offsets of the hashes from the
    recorded sample and the time offsets of the hashes matched in the database for a song.
    The function then returns the size of the largest bin in this histogram as a score.

    :param offsets: List of offset pairs for matching hashes
    :returns: The highest peak in a histogram of time deltas
    :rtype: int
    """
    # Use bins spaced 0.5 seconds apart
    binwidth = 0.5
    tks = list(map(lambda x: x[0] - x[1], offsets))
    hist, _ = np.histogram(tks,
                           bins=np.arange(int(min(tks)),
                                          int(max(tks)) + binwidth + 1,
                                          binwidth))
    return np.max(hist)


def best_match(matches):
    """For a dictionary of song_id: offsets, returns the best song_id.

    Scores each song in the matches dictionary and then returns the song_id with the best score.

    :param matches: Dictionary of song_id to list of offset pairs (db_offset, sample_offset)
       as returned by :func:`~abracadabra.Storage.storage.get_matches`.
    :returns: song_id with the best score.
    :rtype: str
    """
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
                   point_efficiency, target_t, target_f, target_start, db_path, threshold):
    """Recognises a pre-recorded sample.

    Recognises the sample stored at the path ``filename``. The sample can be in any of the
    formats in :data:`recognise.KNOWN_FORMATS`.

    :param filename: Path of file to be recognised.
    :returns: :func:`~abracadabra.recognise.get_song_info` result for matched song or None.
    :rtype: tuple(str, str, str)
    """
    hashes = fingerprint_file(filename=filename, sample_rate=sample_rate,
                              fft_window_size=fft_window_size, peak_box_size=peak_box_size,
                              point_efficiency=point_efficiency, target_t=target_t,
                              target_f=target_f, target_start=target_start)
    matches = get_matches(hashes=hashes, db_path=db_path, threshold=threshold)
    matched_song = best_match(matches=matches)
    info = get_info_for_song_id(song_id=matched_song, db_path=db_path)
    if info is not None:
        return info
    return matched_song


def listen_to_song(filename, format, channels, rate, chunk, record_seconds,
                   sample_rate, fft_window_size, peak_box_size, point_efficiency, target_t, target_f, target_start,
                   db_path, threshold=5):
    """Recognises a song using the microphone.

    Optionally saves the sample recorded using the path provided for use in future tests.
    This function is good for one-off recognitions, to generate a full test suite, look
    into :func:`~abracadabra.record.gen_many_tests`.

    :param filename: The path to store the recorded sample (optional)
    :returns: :func:`~abracadabra.recognise.get_song_info` result for matched song or None.
    :rtype: tuple(str, str, str)
    """
    audio = record_audio(filename=filename, format=format, channels=channels,
                         rate=rate, chunk=chunk, record_seconds=record_seconds)
    hashes = fingerprint_audio(frames=audio, sample_rate=sample_rate, fft_window_size=fft_window_size,
                               peak_box_size=peak_box_size, point_efficiency=point_efficiency,
                               target_t=target_t, target_f=target_f, target_start=target_start)
    matches = get_matches(hashes=hashes, db_path=db_path)
    matched_song = best_match(matches=matches)
    info = get_info_for_song_id(matched_song, db_path=db_path)
    if info is not None:
        return info
    return matched_song
