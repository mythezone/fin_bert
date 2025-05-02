from data.mysql import FileProcessor


import multiprocessing
from task_manager import generate_task_list, get_pending_tasks, log_completed


def worker_init():
    global processor
    processor = FileProcessor()


def worker_task(file_path):
    try:
        processor.process_csv(file_path)
        log_completed(file_path)
        return f"[OK] {file_path}"
    except Exception as e:
        return f"[ERROR] {file_path}: {e}"


def run_parallel_import_pool(
    year, root_dir="/Volumes/ashare/market_data/jydata_unzip", num_processes=20
):
    generate_task_list(root_dir, year)
    pending_files = get_pending_tasks()

    print(f"[INFO] Pending files: {len(pending_files)}")

    with multiprocessing.Pool(processes=num_processes, initializer=worker_init) as pool:
        for result in pool.imap_unordered(worker_task, pending_files):
            print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2014")
    parser.add_argument("--works", type=int, default=10)
    args = parser.parse_args()
    parser.add_argument(
        "--root_dir",
        default="/Volumes/ashare/market_data/jydata_unzip",
        help="Root directory for data files",
    )

    run_parallel_import_pool(args.year, root_dir=args.root_dir, num_processes=args.works)
