# Quick Start (Optimized for T4 GPU (16GB))

## Create new virtual environment, fitting this repository
```
git clone https://github.com/hskim7838/deepfake_detection
cd deepfake_detection
conda create -n deepfake python=3.9 -y
conda activate deepfake
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Download the datasets both training and testing
```
# Create a .json file of your Kaggle account if you don't have your kaggle json file.
mkdir -p ~/.kaggle
vim ~/.kaggle/kaggle.json
# Then, write below contents in the .json file
# {
#   "username": "<Your Username of Kaggle>"
#   "key": "<Token of your Kaggle account>"
# }
chmod 600 ~/.kaggle/kaggle.json

# Before download the training dataset, you need to submit the Google form of Technical University of Munich.
python train_data/download_FF++.py ./train_data -c c23 -t videos -d all --server <server>
python test_data/CDF_v2_download.py
```

## Training
```
python train.py
```
