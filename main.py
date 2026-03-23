from args import get_args
import pandas as pd
import os
from dataset import ObjDetectionDataset
import torch
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Read the dataframes
    train_df = pd.read_csv("CSVs/train_df.csv")
    val_df = pd.read_csv("CSVs/val_df.csv")

    # 2. Prepare dataset
    train_dataset = ObjDetectionDataset(train_df, args)
    val_dataset = ObjDetectionDataset(val_df, args)

    #print(train_df.columns)

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())

    #images, targets = next(iter(train_loader))

    # 4. Initialization the model
    model = build_model(args.backbone, num_classes=args.num_classes + 1)

    # 5. Train the model
    train_model(model, train_loader, val_loader, device)

    

if __name__ == '__main__':
    main()
