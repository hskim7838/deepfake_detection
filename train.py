import argparse
import torch
import torch.nn as nn
from model import EffortWrapper
from src.dataset import get_dataloader
from src.utils import load_config, make_data_list, seed_everything
from transformers import CLIPVisionModel
from tqdm import tqdm

def train(config):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    device = torch.device(config['train']['device'])

    train_list = make_data_list(config, True)
    print(len(train_list))
    train_loader = get_dataloader(
        train_list,
        size=config['data']['size'],
        num_frames=config['data']['num_frames_train'],
        is_training=True,
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        # pin_memory=False,
    )

    # 최초 실행 시에만 하고 지우고 아래 주석 부분 해제하기
    # clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    # clip_model.save_pretrained("./model/clip-vit-large-patch14")

    # local_files_only 부분은 꼭 True로 할 것
    LOCAL_CLIP_PATH = "./model/clip-vit-large-patch14"
    clip_model = CLIPVisionModel.from_pretrained(LOCAL_CLIP_PATH, local_files_only=True)

    model = EffortWrapper(
        clip_model,
        r=config['model']['r'],
        lambda_ksv=config['model']['lambda_ksv']
    )
    model.to(device)
    model.to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="reduce-overhead")
    model = torch.compile(model, mode='default')
    model.train()

    optimizer = torch.optim.AdamW([
        {'params': model.clip_model.parameters(), 'lr': config['train']['lr']},
        {'params': model.classifier.parameters(), 'lr': config['train']['lr'] * 10}
    ], weight_decay=config['train']['weight_decay'])

    criterion = nn.BCEWithLogitsLoss()

    num_epochs = config['train']['epochs']

    for epoch in range(num_epochs):
        total_loss = 0.0
        best_loss = 100.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for frames_or_img, labels, is_video in pbar:
            if frames_or_img.dim() == 5:
                B, N, C, H, W = frames_or_img.shape
                frames_or_img = frames_or_img.view(B * N, C, H, W)
                labels = labels.unsqueeze(1).repeat(1, N).reshape(-1).float()
            else:
                labels = labels.float()
        
        frames_or_img = frames_or_img.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(frames_or_img)
        loss_cls = criterion(logits, labels.unsqueeze(1))

            loss_ksv = model.compute_ksv_loss()
            loss = loss_cls + model.lambda_ksv * loss_ksv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n if pbar.n > 0 else 1))

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"model/model_{epoch}_loss_{total_loss / len(train_loader):.4f}.pt")
        if total_loss/len(train_loader) < best_loss:
            best_loss = total_loss/len(train_loader)
            torch.save(model.state_dict(), f"model/model_best.pt")

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
