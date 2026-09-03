
import os
import glob
import shutil
import kagglehub

def download_job_dataset():
    data_dir = "data/jobs"
    os.makedirs(data_dir, exist_ok=True)
    
    print("--- SmartHire GenAI Job Dataset Downloader ---")

    dataset_handle = "PromptToolkit/naukri-job-dataset" 
    
    print(f"\nDownloading dataset from Kaggle: {dataset_handle}...")
    
    try:
        path = kagglehub.dataset_download(dataset_handle)
        print(f"Dataset downloaded to temporary path: {path}")
        for item in os.listdir(path):
            source_item = os.path.join(path, item)
            target_item = os.path.join(data_dir, item)
            
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, target_item)
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        print(f"\nSuccess! Found CSV files in {data_dir}: {csv_files}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure your kaggle.json file is correctly configured.")

if __name__ == "_main_":
    download_job_dataset()