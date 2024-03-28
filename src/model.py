import sys

sys.path.append("..")
import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torchvision.ops.misc import MLP, Conv2dNormActivation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from collections import OrderedDict
import math
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from processing import create_data_for_vision

from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

import torch.nn.functional as F
import gc

# from focal_loss.focal_loss import FocalLoss
# from torch import Tensor
# import torch.cuda.amp as amp
# import setup
from datetime import datetime
import os


class MultiStationDataset(Dataset):
    def __init__(
        self, dataframes, target, features, past_steps, future_steps, nysm_vars=12
    ):
        """
        dataframes: list of station dataframes like in the SequenceDataset
        target: target error
        features: list of features for model
        sequence_length: int
        """
        self.dataframes = dataframes
        self.features = features
        self.target = target
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.nysm_vars = nysm_vars

    def __len__(self):
        shaper = min(
            [
                self.dataframes[i].values.shape[0]
                - (self.past_steps + self.future_steps)
                for i in range(len(self.dataframes))
            ]
        )
        return shaper

    def __getitem__(self, i):
        # this is the preceeding sequence_length timesteps
        x = torch.stack(
            [
                torch.tensor(
                    dataframe[self.features].values[
                        i : (i + self.past_steps + self.future_steps)
                    ]
                )
                for dataframe in self.dataframes
            ]
        ).to(torch.float32)
        # stacking the sequences from each dataframe along a new axis, so the output is of shape (batch, stations (len(self.dataframes)), past_steps, features)
        y = torch.stack(
            [
                torch.tensor(
                    dataframe[self.target].values[
                        i + self.past_steps : i + self.past_steps + self.future_steps
                    ]
                )
                for dataframe in self.dataframes
            ]
        ).to(torch.float32)
        # this is (batch, stations, future_steps)
        x[-self.future_steps :, : self.nysm_vars] = (
            -999.0
        )  # check that this is setting the right positions to this value
        return x, y


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


def train_model(data_loader, model, optimizer, device, epoch, loss_func):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass and loss computation.
        output = model(X)
        loss = loss_func(output, y)

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()
        gc.collect()

    # Synchronize and aggregate losses in distributed training.
    # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches

    # Print the average loss on the master process (rank 0).
    print("epoch", epoch, "train_loss:", avg_loss)

    return avg_loss


def test_model(data_loader, model, device, epoch, loss_func):
    # Test a deep learning model on a given dataset and compute the test loss.
    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)
        # Forward pass to obtain model predictions.
        output = model(X)
        # Compute loss and add it to the total loss.
        total_loss += loss_func(output, y).item()
        gc.collect()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


def predict(data_loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output


def output(prediction, df_ls):
    import statistics as st

    df_out = pd.DataFrame()
    n = prediction.shape[1]
    i = 0
    print("PREDICT", prediction.shape)
    while n > i:
        target = df_ls[i]["target_error"].tolist()
        # target = target[16:]
        output = prediction[:, i, 0]
        output = output.tolist()
        df_out[f"{i}_transformer_output"] = output
        df_out[f"{i}_target"] = target
        i += 1

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    df_out = df_out.sort_index()
    return df_out


def plot_outputs(df_out, prediction, stations, today_date, clim_div):
    import matplotlib.pyplot as plt

    df_out = df_out.sort_index()
    fig, axs = plt.subplots(
        prediction.shape[1], figsize=(21, 21), sharex=True, sharey=True
    )
    n = prediction.shape[1]
    i = 0
    while n > i:
        axs[i].set_ylabel(f"{stations[i]}")
        axs[i].plot(df_out[f"{i}_target"], c="r", label="Target")
        axs[i].plot(
            df_out[f"{i}_transformer_output"],
            c="b",
            alpha=0.7,
            label="Transformer Output",
        )
        i += 1
    fig.suptitle(f"Transformer Output v Target", fontsize=28)
    axs[-1].set_xticklabels([2018, 2019, 2020, 2021, 2022, 2023], fontsize=18)
    axs[-1].set_xticks(
        np.arange(0, len(df_out["0_target"]), (len(df_out["0_target"])) / 6)
    )
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(
        f"/home/aevans/transformer_ml/src/data/visuals/{today_date}/{clim_div}/output.png"
    )


def eval_model(
    train_loader,
    test_loader,
    model,
    device,
    df_train_ls,
    df_test_ls,
    stations,
    today_date,
    clim_div,
):
    train_predict = predict(train_loader, model, device)
    test_predict = predict(test_loader, model, device)

    train_predict = train_predict.cpu()
    test_predict = test_predict.cpu()

    train_out = output(train_predict, df_train_ls)
    test_out = output(test_predict, df_test_ls)

    dfout = pd.concat([train_out, test_out])
    dfout.to_parquet(
        f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}/{clim_div}/ml_output.parquet"
    )
    plot_outputs(dfout, train_predict, stations, today_date, clim_div)


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


class EarlyStopper:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_time_title(station):
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs(today_date, station)

    return today_date, today_date_hr


def make_dirs(today_date, station):
    if (
        os.path.exists(f"/home/aevans/transformer_ml/src/data/visuals/{today_date}")
        == False
    ):
        os.mkdir(f"/home/aevans/transformer_ml/src/data/visuals/{today_date}")
        os.mkdir(f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}")
    if (
        os.path.exists(
            f"/home/aevans/transformer_ml/src/data/visuals/{today_date}/{station}"
        )
        == False
    ):
        os.mkdir(f"/home/aevans/transformer_ml/src/data/visuals/{today_date}/{station}")
        os.mkdir(f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}/{station}")


def main(EPOCHS, BATCH_SIZE, LEARNING_RATE, CLIM_DIV):
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="transformer_beta",
        workspace="shmaronshmevans",
    )
    torch.manual_seed(101)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    today_date, today_date_hr = get_time_title(CLIM_DIV)
    # create data
    df_train_ls, df_test_ls, features, stations = (
        create_data_for_vision.create_data_for_model(CLIM_DIV, today_date)
    )

    # load datasets
    train_dataset = MultiStationDataset(df_train_ls, "target_error", features, 8, 8)
    test_dataset = MultiStationDataset(df_test_ls, "target_error", features, 8, 8)

    # define model parameters
    model = AaronFormer(
        output_dim=1,
        stations=len(df_train_ls),
        past_timesteps=8,
        future_timesteps=8,
        variables=(len(df_train_ls[0].keys()) - 1),
        num_layers=5,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.05,
        attention_dropout=0.05,
        pos_embedding=0.2,
    )
    if torch.cuda.is_available():
        model.cuda()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # MSE Loss
    loss_func = nn.MSELoss()
    # loss_func = FocalLossV3()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    hyper_params = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "clim_div": str(CLIM_DIV),
    }
    early_stopper = EarlyStopper(20)

    for ix_epoch in range(1, EPOCHS + 1):
        print("Epoch", ix_epoch)
        train_loss = train_model(
            train_loader, model, optimizer, device, ix_epoch, loss_func
        )
        test_loss = test_model(test_loader, model, device, ix_epoch, loss_func)
        print()
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        # if early_stopper.early_stop(test_loss):
        #     print(f"Early stopping at epoch {ix_epoch}")
        #     break

    eval_model(
        train_loader,
        test_loader,
        model,
        device,
        df_train_ls,
        df_test_ls,
        stations,
        today_date,
        CLIM_DIV,
    )
    experiment.end()


main(15, int(200), 7e-4, "Hudson Valley")
