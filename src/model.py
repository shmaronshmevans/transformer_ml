import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import gc
from datetime import datetime
#comet
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model
#pytorch
import torch
import torchvision
import torch.nn as nn
from torchvision.ops.misc import MLP, Conv2dNormActivation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
#ViT
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

# from focal_loss.focal_loss import FocalLoss
# from torch import Tensor
# import torch.cuda.amp as amp
# import setup

# dependencies
from processing import create_data_for_vision
from processing import save_output
from processing import get_time_title


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        pos_embedding: torch.Tensor,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=pos_embedding)
        )  # from BERT
        # self.pos_embedding = nn.Parameter(torch.randn((1 // seq_length) **2, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input += self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        stations: int,
        past_timesteps: int,
        future_timesteps: int,
        num_vars: int,
        pos_embedding: torch.Tensor,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 128,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.future_timesteps = future_timesteps
        self.past_timesteps = past_timesteps
        self.stations = stations
        self.timesteps = future_timesteps + past_timesteps
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.num_vars = num_vars

        self.mlp = torchvision.ops.MLP(
            num_vars, [hidden_dim], None, torch.nn.GELU, dropout=dropout
        )

        seq_length = stations * (future_timesteps + past_timesteps)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            pos_embedding,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if hasattr(self.heads, "pre_logits") and isinstance(
            self.heads.pre_logits, nn.Linear
        ):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in)
            )
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        # n = batch size
        # h = number of stations
        # w = number of time steps
        # c = number of features
        n, h, w, c = x.shape
        torch._assert(
            h == self.stations,
            f"Wrong image height! Expected {self.stations} but got {h}!",
        )
        torch._assert(
            w == self.timesteps,
            f"Wrong image width! Expected {self.timesteps} but got {w}!",
        )

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))

        x = x.reshape(n, h * w, self.num_vars)
        x = self.mlp(x)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" is the future prediction - we will probably just want to select just some of these variables.
        # x \in (batch, stations * timesteps + 1, num_classes = 1)
        x = x[
            :, -(self.stations * self.future_timesteps) :, :
        ]  # this shape is (batch, stations, num_classes = 1)

        x = self.heads(x)  # is a linear transformation from hidden_dim to 1

        x = x.reshape(n, self.stations, self.future_timesteps, self.num_classes)

        return (
            x.squeeze()
        )  # logically we are saying return one value for the each future timestep for each station (interpreted as error)


class AaronFormer(nn.Module):
    def __init__(
        self,
        output_dim,
        stations,
        past_timesteps,
        future_timesteps,
        variables,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        pos_embedding,
    ):
        super().__init__()

        self.encoder = VisionTransformer(
            stations=stations,
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            num_vars=variables,
            pos_embedding=pos_embedding,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=output_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# def focal_loss(output, target, gamma=2.0, alpha=0.25):
#     """Focal loss function.

#     Args:
#         output: The predicted output of the model.
#         target: The ground truth target.
#         gamma: The gamma parameter.
#         alpha: The alpha parameter.

#     Returns:
#         The focal loss.
#     """
#     p = torch.sigmoid(output)
#     pt = p * target + (1 - p) * (1 - target)
#     loss = -alpha * (1 - pt)**gamma * torch.log(pt)
#     return loss

# # version 3: implement wit cpp/cuda to save memory and accelerate
# class FocalSigmoidLossFuncV3(torch.autograd.Function):
#     '''
#     use cpp/cuda to accelerate and shrink memory usage
#     '''
#     @staticmethod
#     @amp.custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, logits, labels, alpha, gamma):
#         #  logits = logits.float()
#         loss = focal_cpp.focalloss_forward(logits, labels, gamma, alpha)
#         ctx.variables = logits, labels, alpha, gamma
#         return loss

#     @staticmethod
#     @amp.custom_bwd
#     def backward(ctx, grad_output):
#         '''
#         compute gradient of focal loss
#         '''
#         logits, labels, alpha, gamma = ctx.variables
#         grads = focal_cpp.focalloss_backward(grad_output, logits, labels, gamma, alpha)
#         return grads, None, None, None

# class FocalLossV3(nn.Module):
#     '''
#     This use better formula to compute the gradient, which has better numeric stability. Also use cuda to shrink memory usage and accelerate.
#     '''
#     def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
#         super(FocalLossV3, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, label):
#         '''
#         Usage is same as nn.BCEWithLogits:
#             >>> criteria = FocalLossV3()
#             >>> logits = torch.randn(8, 19, 384, 384)
#             >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
#             >>> loss = criteria(logits, lbs)
#         '''
#         loss = FocalSigmoidLossFuncV3.apply(logits, label, self.alpha, self.gamma)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         if self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
