import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import CLIPVisionModel, CLIPProcessor
import types

class EffortWrapper(nn.Module):
    def __init__(self, clip_model, r=200, lambda_ksv=0.1):
        super(EffortWrapper, self).__init__()
        self.clip_model = clip_model
        self.r = r
        self.lambda_ksv = lambda_ksv

        self._decompose_weights()

        self.classifier = nn.Linear(1024, 1)

        self.original_total_norm = 0.0
        self.original_delta_w_norm = 0.0
        self._save_original_norms()

    def _decompose_weights(self):
        for name, module in self.clip_model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                out_features, in_features = weight.shape

                U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

                r = min(self.r, len(S) - 1)

                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                W_r = U_r @ torch.diag(S_r) @ Vt_r

                U_tail = U[:, r:]
                S_tail = S[r:]
                Vt_tail = Vt[r:, :]
                Delta_W = U_tail @ torch.diag(S_tail) @ Vt_tail

                with torch.no_grad():
                    module.weight.copy_(W_r + Delta_W)

                self._replace_with_effort_params(module, W_r, Delta_W)

    def _replace_with_effort_params(self, module, W_r, Delta_W):
        out_features, in_features = W_r.shape

        module.Wr = nn.Parameter(W_r, requires_grad=False)
        module.Delta_W = nn.Parameter(Delta_W, requires_grad=True)
        del module.weight

        module.forward = types.MethodType(_effort_linear_forward, module)

    def _save_original_norms(self):
        total_norm = 0.0
        delta_w_norm = 0.0

        for name, module in self.clip_model.named_modules():
            if hasattr(module, 'Wr') and hasattr(module, 'Delta_W'):
                W = module.Wr + module.Delta_W
                total_norm += W.pow(2).sum()
                delta_w_norm += module.Delta_W.pow(2).sum()

        self.original_total_norm = total_norm.detach().clone()
        self.original_delta_w_norm = delta_w_norm.detach().clone()

    def _effort_linear_forward(self, x):
        weight = self.Wr + self.Delta_W
        return nn.functional.linear(x, weight, self.bias)

    def forward(self, x):
        features = self.clip_model(x).pooler_output
        logits = self.classifier(features).squeeze(-1)

        return logits

    def compute_ksv_loss(self):
        current_total_norm = 0.0
        current_delta_w_norm = 0.0

        for name, module in self.clip_model.named_modules():
            if hasattr(module, 'Wr') and hasattr(module, 'Delta_W'):
                W_hat = module.Wr + module.Delta_W
                current_total_norm += W_hat.pow(2).sum()
                current_delta_w_norm += module.Delta_W.pow(2).sum()

        device = current_total_norm.device
        original_total_norm = self.original_total_norm.to(device)
        ksv_loss = (current_total_norm - original_total_norm).abs()

        return ksv_loss

def _effort_linear_forward(self, x):
    weight = self.Wr + self.Delta_W
    return F.linear(x, weight, self.bias)
