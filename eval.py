import argparse
import torch
from src.dataset import get_dataloader
from src.utils import load_config, make_data_list
from model import EffortWrapper
from transformers import CLIPVisionModel
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

def eval_model(config):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    device = torch.device(config['eval']['device'])

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

    val_list = make_data_list(config, False)
    val_loader = get_dataloader(
        val_list,
        size=config['data']['size'],
        num_frames=config['data']['num_frames_test'],
        is_training=False,
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        shuffle=False
    )

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", unit="batch")
        for frames_or_img, labels, is_video in pbar:
            if frames_or_img.dim() == 5:
                B, N, C, H, W = frames_or_img.shape
                frames_or_img = frames_or_img.view(B * N, C, H, W)
                labels = labels.unsqueeze(1).expand(-1, N).reshape(-1)

            frames_or_img = frames_or_img.to(device, non_blocking=True, memory_format=torch.channels_last)
            logits = model(frames_or_img)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()

            if frames_or_img.dim() == 5:
                probs = probs.view(B, N, -1)
                final_probs = probs.mean(dim=1)  # (B, 2)
                final_preds = final_probs.argmax(dim=-1).cpu().numpy()
                final_labels = labels.view(B, N).mean(dim=1).round().cpu().numpy()
            else:
                final_probs = probs
                final_preds = preds
                final_labels = labels.cpu().numpy()

            all_labels.extend(final_labels.tolist())
            all_preds.extend(final_preds.tolist())
            all_probs.extend(final_probs[:, 1].cpu().numpy().tolist())

            if len(all_labels) > 0:
                running_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix(acc=f"{running_acc:.4f}")

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Val Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    with open(config['eval']['output_file'], "w") as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"Val Accuracy: {acc:.4f}\n")
        f.write(f"Val AUC: {auc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_model(config)
