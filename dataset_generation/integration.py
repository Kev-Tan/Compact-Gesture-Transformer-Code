import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from model_utils.configs import TransformerConfig
from model_utils.build_model import build_model
from data_utils.build_dataset import DatasetImgTarget
from data_utils.build_transform import standard_transform
from data_utils.generate_csv import generate_csv

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print("Training batch", batch_idx)
        # inputs shape: (B, F, C, H, W)
        batch_size, frames, channels, height, width = inputs.shape
        
        # Reshape to (B*F, C, H, W) for backbone processing
        inputs = inputs.view(-1, channels, height, width).to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape outputs back to (B, F, num_classes) then average over frames
        outputs = outputs.view(batch_size, frames, -1)  # (4, 21, num_classes)
        outputs = outputs.mean(dim=1)  # (4, num_classes) - average predictions across frames
        
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # if (batch_idx + 1) % 5 == 0:
        print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GestureTransformer")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--out_planes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset_root_path", type=str, default=r"..\..\data\Briareo")
    parser.add_argument("--n_frames", type=int, default=20)
    parser.add_argument("--random_erasing", action="store_true")
    parser.add_argument("--random_horizontal_flip", action="store_true")
    parser.add_argument("--random_cropping", action="store_true")
    parser.add_argument("--restricted_rotation", action="store_true")
    parser.add_argument("--shear", action="store_true")
    parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build config and model
    print("Building model...")
    model = build_model(args)
    model = model.to(device)
    print(f"Model built successfully")
    
    # Build dataset and dataloader
    print("Building dataset...")
    try:
        # Build transform
        transform_function = standard_transform(args, True)
        
        # Build dataset
        train_dataset = DatasetImgTarget(
            root=args.dataset_root_path, 
            split='train', 
            transforms=transform_function,
            n_frames=args.n_frames
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        
        # Test one batch
        sample_batch, sample_label = next(iter(train_loader))
        print(f"Sample batch shape: {sample_batch.shape}")  # (B, F, C, H, W)
        print(f"Sample label shape: {sample_label.shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    print("Starting training...")
    for epoch in range(args.epochs):
        avg_loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("Training complete!")

if __name__ == "__main__":
    main()