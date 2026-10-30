import os
import subprocess
import shutil

def download_datasets():
   
    print("Starting Agricultural Dataset Acquisition")

    # 1. Beans Dataset (iBean via GitHub)
    beans_dir = "./dataset_beans"
    if not os.path.exists(beans_dir):
        print("Cloning iBean dataset (Beans)...")
        try:
            subprocess.run(["git", "clone", "https://github.com/AI-Lab-Makerere/ibean.git", beans_dir], check=True)
            print("Successfully downloaded iBean dataset.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning iBean repository: {e}")
    else:
        print("iBean dataset directory already exists.")

    # Try importing kagglehub for Kaggle datasets
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run 'pip install kagglehub' in your terminal.")
        return

    # 2. Maize/Corn Dataset (Kaggle)
    print("Downloading Corn or Maize Leaf Disease Dataset...")
    try:
        maize_path = kagglehub.dataset_download("smaranjitghose/corn-or-maize-leaf-dataset")
        maize_target = "./dataset_maize"
        if not os.path.exists(maize_target):
            shutil.copytree(maize_path, maize_target)
        print(f"Maize dataset ready at: {maize_target}")
    except Exception as e:
        print(f"Failed to download Maize dataset: {e}")

    # 3. Cassava Dataset (Kaggle)
    print("Downloading Cassava Leaf Disease Classification Dataset...")
    try:
        cassava_path = kagglehub.dataset_download("nirmalsankalana/cassava-leaf-disease-classification")
        cassava_target = "./dataset_cassava"
        if not os.path.exists(cassava_target):
            shutil.copytree(cassava_path, cassava_target)
        print(f"Cassava dataset ready at: {cassava_target}")
    except Exception as e:
        print(f"Failed to download Cassava dataset: {e}")

    print("--- Dataset Acquisition Complete ---")

if __name__ == "__main__":
    download_datasets()