"""
This module provides utilities to load and preprocess the
[Kaggle RSNA Intracranial Hemorrhage Detection Dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection).
"""

from pathlib import Path
from random import random
import polars as pl
from typing import List, Optional
import subprocess
from tqdm import tqdm

COMPETITION_NAME = "rsna-intracranial-hemorrhage-detection"

class DownloadError(Exception):
    """Raised when a DICOM file fails to download from Kaggle CLI."""
    pass

class KaggleRSNAICHDataset:
    def __init__(self,
                 data_path: str,
                 download: bool = False,
                 n_images: Optional[int] = None) -> None:
        self.data_path = data_path
        
        # Create the data directory if it doesn't exist
        Path(self.data_path).mkdir(parents=True, exist_ok=True)

        if download:
            image_ids = self.get_ids()
            if n_images is not None:
                download_ids = pl.Series(image_ids).sample(n=min(n_images, len(image_ids)), with_replacement=False).to_list()
            else:
                download_ids = image_ids
            for image_id in tqdm(download_ids, desc="Downloading DICOM files"):
                self.download(image_id)

        return self

    def get_listing(self) -> pl.DataFrame:
        """
        Returns the list of IDs and labels provided by the competition.
        """
        try:
            lf = pl.scan_csv(f"{self.data_path}/stage_2_train.csv")
            lf = lf.with_columns([
                pl.col("ID").str.split(by="_").list.get(2).alias("HemorrhageType"),
                pl.col("ID").str.split(by="_").list.get(1).alias("ImageID"),
            ])
            train_dataset_listing = lf.collect()
            return train_dataset_listing
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CSV file not found in {self.data_path}. "
                f"Please ensure the dataset is downloaded and placed correctly.\n"
                f"You can download the file 'stage_2_train.csv' from the Kaggle competition page: "
                f"https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data \n"
                f"Then, the file should be placed in the directory: {Path(self.data_path).resolve()}"

            )

    def get_ids(self) -> list[str]:
        """
        Returns a list of all image IDs in the dataset.
        """
        train_dataset_listing = self.get_listing()
        return train_dataset_listing["ImageID"].unique().to_list()

    def download(self,
                 image_id: str,) -> None:
        """
        Download DICOM files from Kaggle CLI.

        Parameters
        ----------
        image_id : str
            DICOM ID to download (without extension).
        """
        remote_file = f"{COMPETITION_NAME}/stage_2_train/ID_{image_id}.dcm"
        command = [
                "kaggle", "competitions", "download",
                "-c", COMPETITION_NAME,
                "-f", remote_file,
                "-p", self.data_path,
            ]
        try:
            subprocess.run(command,
                           check=True,
                           stdout=subprocess.DEVNULL,  # Suppress Kaggle's own progress bar output
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise DownloadError(f"Error downloading {image_id}: {e}")