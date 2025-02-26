import torch
from torch.utils.data import DataLoader
from Source.Models.Detector import VideoTransformer
from Source.Data.Dataset import VideoDataset
from Source.Config.Config import CONFIG
from Source.Utils.logging_config import setup_logging
import os
from datetime import datetime
from tqdm import tqdm

logger = setup_logging('training')


def train_model():
    """Train the video anomaly detection model"""

    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Initialize model
        model = VideoTransformer(
            image_size=CONFIG['data'].image_size,
            patch_size=CONFIG['model'].patch_size,
            num_classes=CONFIG['model'].num_classes,
            dim=CONFIG['model'].dim,
            depth=CONFIG['model'].depth,
            heads=CONFIG['model'].heads,
            mlp_dim=CONFIG['model'].mlp_dim,
            dropout=CONFIG['model'].dropout,
            emb_dropout=CONFIG['model'].emb_dropout,
            num_frames=CONFIG['data'].num_frames
        ).to(device)

        # Initialize datasets
        train_dataset = VideoDataset(
            root_dir=CONFIG['data'].train_path,
            is_train=True
        )

        val_dataset = VideoDataset(
            root_dir=CONFIG['data'].test_path,
            is_train=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['training'].batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['training'].batch_size,
            shuffle=False,
            num_workers=4
        )

        # Initialize optimizer and criterion
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['training'].learning_rate,
            weight_decay=CONFIG['training'].weight_decay
        )

        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0.0

        for epoch in range(CONFIG['training'].num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (videos, labels) in enumerate(tqdm(train_loader)):
                videos, labels = videos.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for videos, labels in val_loader:
                    videos, labels = videos.to(device), labels.to(device)
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100.0 * val_correct / val_total

            # Log results
            logger.info(f"Epoch {epoch + 1}/{CONFIG['training'].num_epochs}")
            logger.info(f"Train Loss: {train_loss / len(train_loader):.4f}")
            logger.info(f"Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss / len(val_loader):.4f}")
            logger.info(f"Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'config': CONFIG,
                    'metadata': {
                        'date_saved': '2025-02-24 12:36:01',
                        'user': 'nguyenlongCS',
                        'val_acc': val_acc,
                        'epoch': epoch
                    }
                }
                torch.save(checkpoint, f"Checkpoints/best_model.pth")
                logger.info(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

    except Exception as e:
        logger.error(f"Training error: {str(e)}")


if __name__ == "__main__":
    train_model()