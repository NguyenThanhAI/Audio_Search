import argparse

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import warnings
warnings.filterwarnings("ignore")

from storage import DataBase
from recognize import register_directory


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--song_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--db_path", type=str, default=None)
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

    song_dir = args.song_dir
    num_workers = args.num_workers
    db_path = args.db_path
    sample_rate = args.sample_rate
    fft_window_size = args.fft_window_size
    peak_box_size = args.peak_box_size
    point_efficiency = args.point_efficiency
    target_t = args.target_t
    target_f = args.target_f
    target_start = args.target_start

    assert 0. < fft_window_size < 1.
    assert 0. < point_efficiency <= 1.

    BaseManager.register("DataBase", DataBase)
    manager = BaseManager()
    manager.start()
    database = manager.DataBase(db_path=db_path)
    database.setup_db()
    register_directory(path=song_dir, num_workers=num_workers, database=database, sample_rate=sample_rate,
                       fft_window_size=fft_window_size, peak_box_size=peak_box_size, point_efficiency=point_efficiency,
                       target_t=target_t, target_f=target_f, target_start=target_start)

    database.close()
