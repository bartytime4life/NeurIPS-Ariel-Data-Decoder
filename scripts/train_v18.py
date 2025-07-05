import torch
from torch.utils.data import DataLoader
from datasets.ariel_dataset import ArielV18_Dataset
from models.vit_student import V18Model
from losses.pinball_loss import PinballLoss
import pandas as pd
import json
import os

with open("configs/v18_config.json") as f:
    cfg = json.load(f)

planet_ids = os.listdir("train")
star_info = pd.read_csv("train_star_info.csv")
dataset = ArielV18_Dataset("train", planet_ids, star_info_df=star_info, is_train=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = V18Model(cfg).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = PinballLoss(cfg["QUANTILES"])

for epoch in range(3):
    model.train()
    for _, x_airs, x_fgs, x_meta, y in loader:
        x_airs, x_meta, y = x_airs.cuda(), x_meta.cuda(), y.cuda()
        out = model(x_airs, x_fgs, x_meta)
        mu = out[..., 0]
        sigma = torch.exp(out[..., 1])
        loss = loss_fn(mu, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
torch.save(model.state_dict(), "model.pt")