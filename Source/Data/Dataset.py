import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
import glob
from Source.Utils.logging_config import setup_logging

logger = setup_logging('data')


class VideoDataset(Dataset):
    """Dataset class for video anomaly detection"""

    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        """
        Initialize the dataset

        Args:
            root_dir: Root directory containing videos
            transform: Optional transforms to apply
            is_train: Whether this is training data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = self._load_dataset()

        self.class_map = {
            'Normal': 0,
            'Abuse': 1,
            'Arrest': 2,
            'Arson': 3,
            'Assault': 4,
            'Burglary': 5,
            'Explosion': 6,
            'Fighting': 7,
            'RoadAccidents': 8,
            'Robbery': 9,
            'Shooting': 10,
            'Shoplifting': 11,
            'Stealing': 12,
            'Vandalism': 13
        }

        logger.info(f"Initialized dataset with {len(self.samples)} samples")

    def _load_dataset(self) -> List[Tuple[str, str]]:
        """Load dataset samples"""
        samples = []
        try:
            if self.is_train:
                # Load anomaly videos
                anomaly_path = os.path.join(self.root_dir, 'Anomaly')
                for class_name in os.listdir(anomaly_path):
                    class_path = os.path.join(anomaly_path, class_name)
                    if os.path.isdir(class_path):
                        for video_file in glob.glob(os.path.join(class_path, '*.mp4')):
                            samples.append((video_file, class_name))

                # Load normal videos
                normal_path = os.path.join(self.root_dir, 'Normal', 'NormalVideos')
                for video_file in glob.glob(os.path.join(normal_path, '*.mp4')):
                    samples.append((video_file, 'Normal'))
            else:
                # Load test data
                for class_name in ['Normal', 'Anomaly']:
                    class_path = os.path.join(self.root_dir, class_name)
                    if os.path.isdir(class_path):
                        for video_file in glob.glob(os.path.join(class_path, '*.mp4')):
                            samples.append((video_file, class_name))

            logger.info(f"Found {len(samples)} videos in {self.root_dir}")
            return samples

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return []

    def _load_video(self, video_path: str, num_frames: int = 16) -> torch.Tensor:
        """Load and preprocess video frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (64, 64))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if len(frames) != num_frames:
                logger.warning(f"Expected {num_frames} frames, got {len(frames)} from {video_path}")
                return None

            frames = np.stack(frames)
            return torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {str(e)}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        try:
            video_path, class_name = self.samples[idx]
            video_tensor = self._load_video(video_path)

            if video_tensor is None:
                # Return a zero tensor if video loading failed
                video_tensor = torch.zeros((3, 16, 64, 64))

            if self.transform:
                video_tensor = self.transform(video_tensor)

            label = self.class_map[class_name]
            return video_tensor, label

        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            # Return a zero tensor and -1 label on error
            return torch.zeros((3, 16, 64, 64)), -1