import os.path
import uuid
import sqlite3
from collections import defaultdict
from contextlib import contextmanager


class DataBase(object):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(database=db_path, check_same_thread=False)

    def setup_db(self):
        cursor = self.conn.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS hash (hash int, offset real, song_id text)")
        cursor.execute("CREATE TABLE IF NOT EXISTS song_info (artist text, album text, title text, song_id text)")
        # dramatically speed up recognition
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON hash (hash)")
        # faster write mode that enables greater concurrency
        # https://sqlite.org/wal.html
        cursor.execute("PRAGMA journal_mode=WAL")
        # reduce load at a checkpoint and reduce chance of a timeout
        cursor.execute("PRAGMA wal_autocheckpoint=300")

        self.conn.commit()
        cursor.close()

    def checkpoint_db(self):
        cursor = self.conn.cursor()

        cursor.execute("PRAGMA wal_checkpoint(FULL)")

        self.conn.commit()
        cursor.close()

    def song_in_db(self, filename):
        cursor = self.conn.cursor()
        song_id = str(uuid.uuid5(uuid.NAMESPACE_OID, os.path.basename(filename)).int)
        cursor.execute("SELECT * FROM song_info WHERE song_id=?", (song_id,))
        song = cursor.fetchone()
        cursor.close()
        return song is not None


    def store_song(self, hashes, song_info):
        if len(hashes) < 1:
            # TODO: After experiments have run, change this to raise error
            # Probably should re-run the peaks finding with higher efficiency
            # or maybe widen the target zone
            return
        cursor = self.conn.cursor()
        cursor.executemany("INSERT INTO hash VALUES (?, ?, ?)", hashes)
        insert_info = [i if i is not None else "Unknown" for i in song_info]
        cursor.execute("INSERT INTO song_info VALUES (?, ?, ?, ?)", (*insert_info, hashes[0][2]))
        self.conn.commit()
        cursor.close()


    def get_matches(self, hashes, threshold=5):
        h_dict = {}
        for h, t, _ in hashes:
            h_dict[h] = t
        in_values = f"({','.join([str(h[0]) for h in hashes])})"
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT hash, offset, song_id FROM hash WHERE hash IN {in_values}")
        results = cursor.fetchall()
        result_dict = defaultdict(list)
        for r in results:
            result_dict[r[2]].append((r[1], h_dict[r[0]]))
        cursor.close()
        return result_dict


    def get_info_for_song_id(self, song_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT artist, album, title FROM song_info WHERE song_id = ?", (song_id,))
        song_info = cursor.fetchone()
        return song_info

    def close(self):
        self.conn.close()
