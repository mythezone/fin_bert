import os
import glob
import pandas as pd


def generate_task_list(
    root_dir: str, year: str, task_file: str = "data/task/"
):
    task_file = os.path.join(task_file, f"task_list_{year}.csv")
    if os.path.exists(task_file):
        print(f"[INFO] Task list already exists: {task_file}")
        return

    pattern = os.path.join(root_dir, year, "**", "*-1M-*.CSV")
    all_files = glob.glob(pattern, recursive=True)

    df = pd.DataFrame({"file_path": all_files})
    df.to_csv(task_file, index=False)
    print(f"[INFO] Task list generated with {len(df)} entries.")


def get_pending_tasks(
    year: str,
    task_file: str = "data/task/",
):
    task_file = os.path.join(task_file, f"task_list_{year}.csv")
    log_file = os.path.join(task_file, f"completed_log_{year}.csv")
    
    df_all = pd.read_csv(task_file)

    if os.path.exists(log_file):
        df_done = pd.read_csv(log_file)
        pending = df_all[~df_all["file_path"].isin(df_done["file_path"])]
    else:
        pending = df_all
    return pending["file_path"].tolist()


def log_completed(file_path: str,year: str,  task_file: str = "data/task/"):
    import csv
    log_file = os.path.join(task_file, f"completed_log_{year}.csv")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([file_path])


if __name__ == "__main__":
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--root_dir",
        type=str,
        default="/app/market_data/jydata_unzip",
        help="Root directory for the data files.",
    )
    arg.add_argument(
        "--year",
        type=str,
        default="2015",
        help="Year for the data files.",
    )
    args = arg.parse_args()
    generate_task_list(root_dir=args.root_dir, year=args.year)
