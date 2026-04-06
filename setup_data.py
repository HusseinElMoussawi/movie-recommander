"""
setup_data.py — Downloads and extracts the MovieLens small dataset.
Run once before training: python setup_data.py
"""

import urllib.request
import zipfile
import os
import shutil

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = "ml-latest-small.zip"
DATA_DIR = "data"


def download_and_extract():
    print("📥 Downloading MovieLens dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)

    print("📦 Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(".")

    # Move CSVs to /data
    os.makedirs(DATA_DIR, exist_ok=True)
    extracted_dir = "ml-latest-small"
    for fname in ["ratings.csv", "movies.csv", "links.csv", "tags.csv"]:
        src = os.path.join(extracted_dir, fname)
        dst = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Cleanup
    os.remove(ZIP_PATH)
    shutil.rmtree(extracted_dir, ignore_errors=True)
    print(f"✅ Data ready in '{DATA_DIR}/'")


if __name__ == "__main__":
    download_and_extract()
