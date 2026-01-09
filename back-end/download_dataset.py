"""
Video Game Reviews Dataset Download Script
Downloads the latest video game reviews and ratings dataset from Kaggle.
"""

import kagglehub
import pandas as pd
import os
from pathlib import Path

def download_video_game_dataset():
    """
    Download the latest video game reviews and ratings dataset from Kaggle.
    
    Returns:
        str: Path to the downloaded dataset files
    """
    try:
        print("Starting download of video game reviews dataset...")
        
        # Download latest version
        path = kagglehub.dataset_download("jahnavipaliwal/video-game-reviews-and-ratings")
        
        print(f"Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        # List the files in the dataset
        dataset_files = list(Path(path).glob("*"))
        print(f"\nDataset contains {len(dataset_files)} files:")
        for file in dataset_files:
            print(f"  - {file.name}")
        
        return str(path)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_dataset(dataset_path: str):
    """
    Load the video game reviews dataset into pandas DataFrames.
    
    Args:
        dataset_path: Path to the downloaded dataset
        
    Returns:
        dict: Dictionary containing loaded DataFrames
    """
    try:
        dataset_path = Path(dataset_path)
        dataframes = {}
        
        # Find and load CSV files
        csv_files = list(dataset_path.glob("*.csv"))
        
        for csv_file in csv_files:
            print(f"Loading {video_game_reviews.csv}...")
            df = pd.read_csv(csv_file)
            dataframes[csv_file.stem] = df
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {list(df.columns)}")
            print()
        
        return dataframes
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

def explore_dataset(dataframes: dict):
    """
    Explore the loaded dataset to understand its structure.
    
    Args:
        dataframes: Dictionary containing loaded DataFrames
    """
    print("=== Dataset Exploration ===")
    
    for name, df in dataframes.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:")
        print(df.dtypes)
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nBasic statistics:")
        print(df.describe())

if __name__ == "__main__":
    # Download the dataset
    dataset_path = download_video_game_dataset()
    
    if dataset_path:
        # Load the dataset
        dataframes = load_dataset(dataset_path)
        
        if dataframes:
            # Explore the dataset
            explore_dataset(dataframes)
            
            # Save the dataset path for later use
            with open("dataset_path.txt", "w") as f:
                f.write(dataset_path)
            print(f"\nDataset path saved to dataset_path.txt")
        else:
            print("Failed to load dataset files.")
    else:
        print("Failed to download dataset.")
