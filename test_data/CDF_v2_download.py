import os
import zipfile
from pathlib import Path
import kaggle

def download_kaggle_dataset(dataset_slug, download_path="./dataset", unzip=True):
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_slug}")
    print(f"Target path: {download_path.absolute()}")
    
    kaggle.api.dataset_download_files(
        dataset_slug, 
        path=str(download_path), 
        unzip=False,
        force=True
    )
    
    print("Download completed (.zip files)")
    
    if unzip:
        print("Extracting zip files...")
        for zip_file in download_path.glob("*.zip"):
            print(f"Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            zip_file.unlink()
            print(f"Completed: {zip_file.name}")
    
    print(f"All done: {download_path.absolute()}")

if __name__ == "__main__":
    dataset_slug = "lephanminhkhoa/deepfake-benchmark-cdf-v2"
    target_path = "./test_data/CDF_v2"
    
    download_kaggle_dataset(
        dataset_slug=dataset_slug,
        download_path=target_path,
        unzip=False
    )
    
    print(f"Final dataset path: {target_path}")
