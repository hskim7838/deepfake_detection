import argparse
import yaml
import torch
import pandas as pd
from src.dataset import get_dataloader
from src.utils import load_config, make_data_list
from model import EffortWrapper
from transformers import CLIPVisionModel
from tqdm import tqdm

def inference(config):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    device = torch.device(config['train']['device'])

    LOCAL_CLIP_PATH = "model/clip-vit-large-patch14"
    clip_model = CLIPVisionModel.from_pretrained(LOCAL_CLIP_PATH, local_files_only=True)

    model = EffortWrapper(
        clip_model,
        r=config['model']['r'],
        lambda_ksv=config['model']['lambda_ksv']
    )
    model.load_state_dict(torch.load("model/model.pt"))
    model.to(device)
    model.to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="reduce-overhead")
    model = torch.compile(model, mode='default')
    model.eval()

    test_list = make_data_list(config, False)
    test_loader = get_dataloader(
        test_list,
        size=config['data']['size'],
        num_frames=config['data']['num_frames_test'],
        is_training=False,
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        shuffle=False
    )

    all_ids = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Inferencing", unit="batch")
        for frames_or_img, labels, is_video in pbar:
            if frames_or_img.dim() == 5:
                B, N, C, H, W = frames_or_img.shape
                frames_or_img = frames_or_img.view(B * N, C, H, W)
                labels = labels.unsqueeze(1).expand(-1, N).reshape(-1)

            frames_or_img = frames_or_img.to(device, non_blocking=True, memory_format=torch.channels_last)
            logits = model(frames_or_img)
            probs = torch.softmax(logits, dim=-1)

            if frames_or_img.dim() == 5:
                probs = probs.view(B, N, -1)
                final_probs = probs.mean(dim=1)
                preds = final_probs.argmax(dim=-1).cpu().numpy()
            else:
                preds = probs.argmax(dim=-1).cpu().numpy()

            all_ids.extend([str(i) for i in range(len(preds))])
            all_preds.extend(preds.tolist())

    df = pd.DataFrame({"id": all_ids, "label": all_preds})
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    inference(config)
