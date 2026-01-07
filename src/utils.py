import yaml
from pathlib import Path
import torch
import random
import numpy as np
import logging

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def make_data_list_(data_dir):
    data_list = []
    data_dir = Path(data_dir)

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.webp'}
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

    real_dir = data_dir / "real"
    if real_dir.exists():
        for ext in IMAGE_EXTS:
            data_list.extend([(p, 0) for p in real_dir.glob(f"*{ext}")])
        for ext in VIDEO_EXTS:
            data_list.extend([(p, 0) for p in real_dir.glob(f"*{ext}")])

    fake_dir = data_dir / "fake"
    if fake_dir.exists():
        for ext in IMAGE_EXTS:
            data_list.extend([(p, 1) for p in fake_dir.glob(f"*{ext}")])
        for ext in VIDEO_EXTS:
            data_list.extend([(p, 1) for p in fake_dir.glob(f"*{ext}")])

    return data_list

def make_data_list(config, is_training=True):
    data_list = []
    
    if is_training:
        print("Loading FF++ training data...")
        base_path = Path(config['data']['train_dir'])
        print(f"The path of train directory: {base_path}")
        compression = "c23"

        real_dir = f"{base_path}/original_sequences/youtube/{compression}/videos/"
        real_dir = Path(real_dir)
        for video_path in real_dir.glob("*.mp4"):
            data_list.append((str(video_path), 0))
        
        fake_dir = f"{base_path}/manuipulated_sequences/Deepfakes/{compression}/videos"
        fake_dir = Path(fake_dir)
        for video_path in fake_dir.glob("*.mp4"):
            data_list.append((str(video_path), 1))

        n_repeat = config['train']['n_repeat']
        data_list *= n_repeat

        return data_list

    else:
        data_file = config['data']['test_file']
        data_root = config['data']['test_dir']
        if dataset == 'CDF-v2':
            print("Loading CDF-v2 data...")

            with open(os.path.join(data_root, data_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    path, label = line.strip().split()
                    full_path = os.path.join(data_root, path)
                    
                    data_list.append((full_path, int(label)))

        print(f"Test: {len(data_list)} samples")
    return data_list

