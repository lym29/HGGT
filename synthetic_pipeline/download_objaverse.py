import multiprocessing
import argparse
from pathlib import Path
from utils.objaverse import DownloadObjaverse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Objaverse assets used by the GraspXL object list."
    )
    parser.add_argument(
        "--uid_file",
        type=Path,
        required=True,
        help="Path to graspxl_obj_ids.txt or another newline-separated UID list.",
    )
    parser.add_argument(
        "--download_path",
        type=Path,
        required=True,
        help="Directory where Objaverse assets will be downloaded.",
    )
    parser.add_argument(
        "--download_processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel download processes.",
    )
    return parser.parse_args()


def load_uids(uid_file):
    with uid_file.open("r") as file_list:
        return sorted(line.strip() for line in file_list if line.strip())


def main():
    args = parse_args()
    uids = load_uids(args.uid_file)

    down_objaverse = DownloadObjaverse(download_path=str(args.download_path))
    objects = down_objaverse.load_objects(
        uids=uids,
        download_processes=args.download_processes,
    )

    print("FINISH! Downloaded", len(objects), "objects")


if __name__ == "__main__":
    main()
