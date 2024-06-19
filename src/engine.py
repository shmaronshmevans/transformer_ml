import sys

sys.path.append("..")

import pandas as pd
import numpy as np
import gc
from datetime import datetime
#comet
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

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

import gc

import model2

#from processing import create_data_for_vision
from processing import get_time_title
from processing import save_output_images
from processing import read_in_images

# class MultiStationDataset(Dataset):
#     def __init__(
#         self, dataframes, target, features, past_steps, future_steps, nysm_vars=14
#     ):
#         """
#         dataframes: list of station dataframes like in the SequenceDataset
#         target: target error
#         features: list of features for model
#         sequence_length: int
#         """
#         self.dataframes = dataframes
#         self.features = features
#         self.target = target
#         self.past_steps = past_steps
#         self.future_steps = future_steps
#         self.nysm_vars = nysm_vars

#     def __len__(self):
#         shaper = min(
#             [
#                 self.dataframes[i].values.shape[0]
#                 - (self.past_steps + self.future_steps)
#                 for i in range(len(self.dataframes))
#             ]
#         )
#         return shaper

#     def __getitem__(self, i):
#         # this is the preceeding sequence_length timesteps
#         x = torch.stack(
#             [
#                 torch.tensor(
#                     dataframe[self.features].values[
#                         i : (i + self.past_steps + self.future_steps)
#                     ]
#                 )
#                 for dataframe in self.dataframes
#             ]
#         ).to(torch.float32)

#         # stacking the sequences from each dataframe along a new axis, so the output is of shape (batch, stations (len(self.dataframes)), past_steps, features)
#         y = torch.stack(
#             [
#                 torch.tensor(
#                     dataframe[self.target].values[
#                         i + self.past_steps : i + self.past_steps + self.future_steps
#                     ]
#                 )
#                 for dataframe in self.dataframes
#             ]
#         ).to(torch.float32)

#         # this is (stations, seq_len, features)
#         # Assuming 'self.future_steps' and 'self.nysm_vars' are defined properly, and 'x' has the appropriate dimensions

#         #persistence edit
#         x[:, -self.future_steps:, -self.nysm_vars:] = x[:, self.future_steps:self.future_steps, -self.nysm_vars:].expand(-1, self.future_steps, -1)

#         #-999 edit
#         # x[:, -self.future_steps:, -self.nysm_vars:] = -999
#         return x, y

# option 2
# class MultiStationDataset(Dataset):
#     def __init__(self, dataframes, target, features, past_steps, future_steps, nysm_vars=14):
#         """
#         dataframes: list of station dataframes like in the SequenceDataset
#         target: target error
#         features: list of features for model
#         past_steps: number of past timesteps to use as input
#         future_steps: number of future timesteps to predict
#         nysm_vars: number of NYSM variables
#         """
#         self.dataframes = dataframes
#         self.features = features
#         self.target = target
#         self.past_steps = past_steps
#         self.future_steps = future_steps
#         self.nysm_vars = nysm_vars

#     def __len__(self):
#         shaper = min(
#             [
#                 len(df) - (self.past_steps + self.future_steps)
#                 for df in self.dataframes
#             ]
#         )
#         return shaper

#     def __getitem__(self, i):
#         # Create x (features) tensor from past_steps
#         x = torch.stack(
#             [
#                 torch.tensor(
#                     df[self.features].iloc[i : i + self.past_steps].values
#                 )
#                 for df in self.dataframes
#             ]
#         ).float()

#         # Create y (target) tensor from future_steps
#         y = torch.stack(
#             [
#                 torch.tensor(
#                     df[self.target].iloc[i + self.past_steps : i + self.past_steps + self.future_steps].values
#                 )
#                 for df in self.dataframes
#             ]
#         ).float()

#         # Optionally, set the last future_steps rows of the last nysm_vars columns in x to a specific value if needed
#         # Uncomment the line below if you want to modify x as per the previous requirement
#         # x[:, -self.future_steps:, -self.nysm_vars:] = x[:, self.future_steps:self.future_steps + 1, -self.nysm_vars:].expand(-1, self.future_steps, -1)
#         # x[:, -self.future_steps:, -self.nysm_vars:] = -999

#         return x, y

#option 3
class ImageSequenceDataset(Dataset):
    def __init__(self, image_list, dataframe, target, sequence_length, transform=None):
        self.image_list = image_list
        self.dataframe = dataframe
        self.transform = transform
        self.sequence_length = sequence_length
        self.target = target

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        images = []
        i_start = max(0, i - self.sequence_length + 1)

        for j in range(i_start, i + 1):
            if j < len(self.image_list):
                img_name = self.image_list[j]
                image = np.load(img_name).astype(np.float32)
                image = image[:, :, 4:]
                if self.transform:
                    image = self.transform(image)
                images.append(torch.tensor(image))
            else:
                pad_image = torch.zeros_like(images[0])
                images.append(pad_image)

        while len(images) < self.sequence_length:
            pad_image = torch.zeros_like(images[0])
            images.insert(0, pad_image)

        images = torch.stack(images)
        images = images.to(torch.float32)

        # Extract target values
        y = self.dataframe[self.target].values[i_start : i + 1]
        if len(y) < self.sequence_length:
            pad_width = (self.sequence_length - len(y), 0)
            y = np.pad(y, (pad_width, (0, 0)), "constant", constant_values=0)

        y = torch.tensor(y).to(torch.float32)
        return images, y



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

def train_model(data_loader, model, optimizer, device, epoch, loss_func, accumulation_steps=4):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    optimizer.zero_grad()
    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass and loss computation.
        output = model(X)
        loss = loss_func(output[-1], y[-1])

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Accumulate gradients and perform optimization step
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()

        gc.collect()
        # Clear CUDA cache (optional, use only if necessary)
        torch.cuda.empty_cache()

    # Perform the final optimization step if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

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
        total_loss += loss_func(output[-1], y[-1]).item()
        gc.collect()
        # Clear CUDA cache (optional, use only if necessary)
        torch.cuda.empty_cache()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss

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

def main(EPOCHS, BATCH_SIZE, LEARNING_RATE, CLIM_DIV, past_timesteps, forecast_hour, single):
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="transformer_beta",
        workspace="shmaronshmevans",
    )
    torch.manual_seed(101)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    today_date, today_date_hr = get_time_title.get_time_title(CLIM_DIV)
    # create data
    # df_train_ls, df_test_ls, features, stations = (
    #     create_data_for_vision.create_data_for_model(CLIM_DIV, today_date, forecast_hour, single)
    # )
    train_df, test_df, train_ims, test_ims, target, stations = read_in_images.create_data_for_model(CLIM_DIV)

    # load datasets
    # train_dataset = MultiStationDataset(df_train_ls, 'target_error', features, past_timesteps, forecast_hour)
    # test_dataset = MultiStationDataset(df_test_ls, 'target_error', features, past_timesteps, forecast_hour)

    train_dataset = ImageSequenceDataset(train_ims, train_df, target, past_timesteps)
    test_dataset = ImageSequenceDataset(test_ims, test_df, target, past_timesteps)

    # define model parameters
    ml = model2.AaronFormer(
        output_dim=1,
        stations=6,
        past_timesteps=past_timesteps,
        future_timesteps=forecast_hour,
        variables=25,
        num_layers=1,
        num_heads=12,
        hidden_dim=252,
        mlp_dim=1032,
        dropout=0.1,
        attention_dropout=0.2,
        pos_embedding=0.02,
        time_embedding = 0.02
    )
    if torch.cuda.is_available():
        ml.cuda()

    # Adam Optimizer
    optimizer = torch.optim.AdamW(ml.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # MSE Loss
    loss_func = nn.MSELoss()
    # loss_func = FocalLossV3()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
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
        "forecast_hour": forecast_hour
    }
    early_stopper = EarlyStopper(20)

    for ix_epoch in range(1, EPOCHS + 1):
        print("Epoch", ix_epoch)
        train_loss = train_model(
            train_loader, ml, optimizer, device, ix_epoch, loss_func
        )
        test_loss = test_model(test_loader, ml, device, ix_epoch, loss_func)
        print()
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        scheduler.step(test_loss)
        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at epoch {ix_epoch}")
            break

    save_output_images.eval_model(
        train_loader,
        test_loader,
        ml,
        device,
        target,
        train_df,
        test_df,
        stations,
        today_date,
        today_date_hr,
        CLIM_DIV,
        forecast_hour, 
        past_timesteps,
    )
    experiment.end()


main(EPOCHS= 100, 
BATCH_SIZE= int(500), 
LEARNING_RATE= 2e-5, 
CLIM_DIV= "Mohawk Valley", 
past_timesteps= 4, 
forecast_hour= 4, 
single=False)