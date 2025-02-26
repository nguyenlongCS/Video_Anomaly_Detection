import torch
import os
from datetime import datetime
from Source.Models.Detector import VideoTransformer
from Source.Config.Config import CONFIG


def init_model_checkpoints():
    # Set current time and user
    current_time = "2025-02-24 11:58:42"
    current_user = "nguyenlongCS"

    print(f"Initializing model checkpoints...")
    print(f"Current Date and Time (UTC): {current_time}")
    print(f"Current User: {current_user}")

    # Create Checkpoints directory if it doesn't exist
    os.makedirs("Checkpoints", exist_ok=True)

    # Model configurations
    model_types = ["vit", "timesformer", "videoswintransformer"]

    for model_type in model_types:
        try:
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
            )

            # Create checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': {
                    'image_size': CONFIG['data'].image_size,
                    'patch_size': CONFIG['model'].patch_size,
                    'num_classes': CONFIG['model'].num_classes,
                    'dim': CONFIG['model'].dim,
                    'depth': CONFIG['model'].depth,
                    'heads': CONFIG['model'].heads,
                    'mlp_dim': CONFIG['model'].mlp_dim,
                    'dropout': CONFIG['model'].dropout,
                    'emb_dropout': CONFIG['model'].emb_dropout,
                    'num_frames': CONFIG['data'].num_frames
                },
                'metadata': {
                    'date_saved': current_time,
                    'user': current_user,
                    'val_acc': 0.0,
                    'epoch': 0
                }
            }

            # Save checkpoint
            checkpoint_path = f"Checkpoints/{model_type}_model.pth"
            torch.save(checkpoint, checkpoint_path)

            print(f"Successfully saved {model_type} model checkpoint:")
            print(f"- Path: {checkpoint_path}")
            print(f"- Date: {current_time}")
            print(f"- User: {current_user}")
            print("-" * 50)

        except Exception as e:
            print(f"Error initializing {model_type} model: {str(e)}")


if __name__ == "__main__":
    init_model_checkpoints()