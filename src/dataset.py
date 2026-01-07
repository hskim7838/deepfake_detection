import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.webp'}

def data_load(path, size, num_frames=8, is_training=True):
    path = str(path)
    ext = '.' + path.split('.')[-1].lower()
    is_video = ext not in IMAGE_EXTS

    if not is_video:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {path}")

        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, False

    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_frames = num_frames if is_training else 32

        frame_indices = np.linspace(0, length - 1, target_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (size, size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, size, size, dtype=torch.float32))

        cap.release()

        video_frames = torch.stack(frames, dim=0)
        return video_frames, True

class ForgeryDataset(Dataset):
    def __init__(self, data_list, size=224, num_frames=8, is_training=True):
        
        self.data_list = data_list
        self.size = size
        self.num_frames = num_frames
        self.is_training = is_training

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        frames_or_img, is_video = data_load(path, self.size, self.num_frames, self.is_training)
        return frames_or_img, label, is_video

def get_dataloader(data_list, size=224, num_frames=8, is_training=False, batch_size=4, num_workers=8):
    dataset = ForgeryDataset(data_list, size, num_frames, is_training)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader
