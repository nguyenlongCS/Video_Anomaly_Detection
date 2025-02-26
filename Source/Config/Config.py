from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_size: int = 64
    patch_size: int = 8
    num_classes: int = 14  # Normal + 13 anomaly types
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    emb_dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 100000


@dataclass
class DataConfig:
    train_path: str = "Data/Train"
    test_path: str = "Data/Test"
    image_size: int = 64
    num_frames: int = 16
    frame_interval: int = 2


# Global configuration
CONFIG = {
    'model': ModelConfig(),
    'training': TrainingConfig(),
    'data': DataConfig(),
    'metadata': {
        'created_by': 'nguyenlongCS',
        'created_at': '2025-02-24 12:36:01',
        'version': '1.0.0'
    }
}