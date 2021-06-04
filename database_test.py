import os
import time
import sqlite3
import numpy as np
from fingerprint import fingerprint_file, my_spectrogram, find_peaks, idxs_to_tf_pairs, file_to_spectrogram, read_audio_file, fingerprint_audio
from recognize import recognize_song, best_match
from storage import DataBase
import uuid

conn = sqlite3.connect("music_3_6_2021.sqlite")
cursor = conn.cursor()

#cursor.execute("SELECT * FROM hash")
#records = cursor.fetchall()
#cursor.close()
#
#for record in records:
#    print(record)
#
#print("===============================================================================")
#
#cursor = conn.cursor()
#cursor.execute("SELECT * FROM song_info")
#records = cursor.fetchall()
#cursor.close()
#
#for record in records:
#    print(record)

#song = recognize_song(filename=r"D:\1NamNguyenTheMinh.mp3",
#                       db_path="music.sqlite", sample_rate=44100,
#                       fft_window_size=0.2, peak_box_size=30, point_efficiency=0.8, target_t=1.8,
#                       target_f=4000, target_start=0.05, threshold=5)
database = DataBase(db_path=r"D:\database_class.sqlite")
audio, sr = read_audio_file(audio_path=r"D:\Music\Han\Advice.mp3", sr_desired=44100)
print(np.min(audio), np.max(audio))
audio += 0.05 * np.random.randn(audio.shape[0])
start = time.time()
hashes = fingerprint_audio(audio, sample_rate=44100,
                         fft_window_size=0.2, peak_box_size=30, point_efficiency=0.8, target_t=1.8,
                         target_f=4000, target_start=0.05)
matches = database.get_matches(hashes=hashes, threshold=5)
matched_song = best_match(matches=matches)
song = database.get_info_for_song_id(song_id=matched_song)
database.close()
end = time.time()
print(song, end - start)

cursor = conn.cursor()
cursor.execute("SELECT * FROM hash WHERE song_id = ?", (str(uuid.uuid5(uuid.NAMESPACE_OID, os.path.basename(r"D:\Music\Han\Advice.mp3")).int),))
song_hashes = cursor.fetchall()
song_hashes.sort(key=lambda x: x[1])
cursor.close()
print(len(song_hashes))

conn_1 = sqlite3.connect(r"D:\database_class.sqlite")

cursor_1 = conn_1.cursor()
cursor_1.execute("SELECT * FROM hash WHERE song_id = ?", (str(uuid.uuid5(uuid.NAMESPACE_OID, os.path.basename(r"D:\Music\Han\Advice.mp3")).int),))
song_hashes = cursor_1.fetchall()
song_hashes.sort(key=lambda x: x[1])
cursor_1.close()
print(len(song_hashes))

hashes = fingerprint_file(filename=r"D:\Music\Han\Advice.mp3", sample_rate=44100, fft_window_size=0.2, peak_box_size=30,
                            point_efficiency=0.8, target_t=1.8, target_f=4000, target_start=0.05)
hashes.sort(key=lambda x: x[1])
print(len(hashes))
#hashes_1 = fingerprint_file(filename=r"D:\Music\Han\Advice.mp3", sample_rate=44100, fft_window_size=0.2, peak_box_size=20,
#                            point_efficiency=0.8, target_t=1.8, target_f=4000, target_start=0.05)
#hashes_2 = fingerprint_file(filename=r"D:\Advice.mp3", sample_rate=44100, fft_window_size=0.2, peak_box_size=20,
#                            point_efficiency=0.8, target_t=1.8, target_f=4000, target_start=0.05)
#print("hashes_1: {}".format(hashes_1))
#print("hashes_2: {}".format(hashes_2))

#f, t, Sxx = file_to_spectrogram(filename=r"D:\Music\Han\Advice.mp3", sample_rate=44100, fft_window_size=0.2)
#peaks = find_peaks(Sxx=Sxx, peak_box_size=20, point_efficiency=0.8)
#peaks.sort(key=lambda x: x[1])
##peaks = idxs_to_tf_pairs(idxs=peaks, t=t, f=f)
#
#print("peaks_1: {}".format(peaks))


#f, t, Sxx = file_to_spectrogram(filename=r"D:\Advice.mp3", sample_rate=44100, fft_window_size=0.2)
#peaks = find_peaks(Sxx=Sxx, peak_box_size=20, point_efficiency=0.8)
#peaks.sort(key=lambda x: x[1])
##peaks = idxs_to_tf_pairs(idxs=peaks, t=t, f=f)
#
#print("peaks_2: {}".format(peaks))