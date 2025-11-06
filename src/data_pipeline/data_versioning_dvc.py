"""
Module: data_versioning_dvc.py
===============================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This script provides a function to track and version raw or processed data files using DVC (Data Version Control)
and Git. It supports remote storage configuration (e.g., Supabase, S3) with credentials loaded from 
environment variables.

Features:
- Initializes DVC in the repository if not already initialized.
- Configures a remote storage if it doesn't exist.
- Adds a specified file to DVC tracking.
- Adds the corresponding .dvc file to Git and commits the change.
- Pushes the data to the configured DVC remote.
"""
# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import subprocess
import os
from dotenv import load_dotenv

from src.utils.const import (RAW_DATA_DIR,raw_file_name)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def dvc_track_processed_file(logger,file_path: str) -> None:
    """
    Track a file using DVC, commit the .dvc file to Git, and push to remote storage.

    Args:
        file_path (str): Path to the file to be tracked and pushed.

    Raises:
        ValueError: If remote_name is missing or invalid.
        subprocess.CalledProcessError: If any DVC or Git command fails.
    """
    # Load credentials and remote details from environment variables
    # Load credentials from environment
    load_dotenv()
    remote_url = os.getenv("DVC_REMOTE_URL")
    access_key = os.getenv("DVC_ACCESS_KEY")
    secret_key = os.getenv("DVC_SECRET_KEY")
    endpointurl = os.getenv("DVC_ENDPOINT")
    remote_name = "supabase"
    if None not in [remote_url, access_key, secret_key, endpointurl]:
        try:
            # Initialize DVC if not already
            if not os.path.exists(".dvc"):
                logger.info("Initializing DVC...")
                subprocess.run(["dvc", "init"], check=True)

            # List existing remotes
            remotes = subprocess.run(
                ["dvc", "remote", "list"], capture_output=True, text=True, check=True
            )
            remotes_stdout = remotes.stdout or ""
            logger.info(f"Existing DVC remotes:\n{remotes_stdout}")

            if remote_name not in remotes_stdout:
                logger.info(f"Setting up DVC remote '{remote_name}'...")
                # Add remote using alias, not URL as the name
                subprocess.run(["dvc", "remote", "add", "-d", remote_name, remote_url], check=True)
                subprocess.run(["dvc", "remote", "modify", remote_name, "access_key_id", access_key, "--local"], check=True)
                subprocess.run(["dvc", "remote", "modify", remote_name, "secret_access_key", secret_key, "--local"], check=True)
                subprocess.run(["dvc", "remote", "modify", remote_name, "endpointurl", endpointurl, "--local"], check=True)

            # Add file to DVC
            subprocess.run(["dvc", "add", file_path], check=True)

            # Add the .dvc file to Git
            #subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)

            # Commit changes
            #subprocess.run(["git", "commit", "-m", f"Track {file_path} with DVC"], check=True)

            # Push to remote
            subprocess.run(["dvc", "push"], check=True)

            logger.info(f"{file_path} tracked successfully with DVC and pushed.")

        except subprocess.CalledProcessError as e:
            logger.info(f"Error tracking {file_path}: {e}")
    else:
        logger.info(f"Setup the requirements in .env to  tracked your data with DVC.")



if __name__ == "__main__":
    # Usage
    RAW_DATA_PATH = f"{RAW_DATA_DIR}/{raw_file_name}"
    dvc_track_processed_file(logger,RAW_DATA_PATH)
