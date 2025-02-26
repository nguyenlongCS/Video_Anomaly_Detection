import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List, Tuple, Optional, Union
from datetime import datetime
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


class VideoPreprocessor:
    """
    A class for preprocessing video data for anomaly detection.

    Created by: nguyenlongCS
    Date: 2025-02-24 11:37:06 UTC
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (64, 64),
                 num_frames: int = 16,
                 normalize: bool = True,
                 augment: bool = True):
        """
        Initialize the video preprocessor.

        Args:
            target_size: Tuple of (height, width) for resizing frames
            num_frames: Number of frames to extract from each video
            normalize: Whether to normalize pixel values
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.num_frames = num_frames
        self.normalize = normalize
        self.augment = augment

        # Define normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Initialize transformations
        self._setup_transforms()

        logging.info(f"Initialized VideoPreprocessor with target size {target_size}")

    def _setup_transforms(self):
        """Setup data augmentation and normalization transforms"""
        transform_list = []

        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            ])

        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            )

        self.transform = transforms.Compose(transform_list)

    def extract_frames(self,
                       video_path: str,
                       uniform_sampling: bool = True) -> Optional[np.ndarray]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file
            uniform_sampling: Whether to sample frames uniformly or sequentially

        Returns:
            numpy array of frames or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                logging.error(f"No frames found in video: {video_path}")
                return None

            if uniform_sampling:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                frame_indices = np.arange(min(self.num_frames, total_frames))

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    frame = cv2.resize(frame, self.target_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    logging.warning(f"Failed to read frame {idx} from {video_path}")

            cap.release()

            if len(frames) < self.num_frames:
                logging.warning(
                    f"Only extracted {len(frames)} frames from {video_path}, "
                    f"expected {self.num_frames}"
                )
                return None

            return np.stack(frames)

        except Exception as e:
            logging.error(f"Error processing video {video_path}: {str(e)}")
            return None

    def preprocess_video(self,
                         video_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess a video file for model input.

        Args:
            video_path: Path to the video file

        Returns:
            Preprocessed video tensor or None if preprocessing fails
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        if frames is None:
            return None

        # Convert to tensor
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

        # Apply transforms
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor

    def preprocess_batch(self,
                         video_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Preprocess a batch of videos.

        Args:
            video_paths: List of paths to video files

        Returns:
            Tuple of (batch tensor, list of successfully processed paths)
        """
        processed_videos = []
        successful_paths = []

        for video_path in video_paths:
            video_tensor = self.preprocess_video(video_path)
            if video_tensor is not None:
                processed_videos.append(video_tensor)
                successful_paths.append(video_path)

        if not processed_videos:
            return None, []

        return torch.stack(processed_videos), successful_paths

    def preprocess_frame(self,
                         frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Preprocessed frame tensor
        """
        # Resize frame
        frame = cv2.resize(frame, self.target_size)

        # Convert to RGB if needed
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:  # BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Apply transforms
        if self.transform:
            frame_tensor = self.transform(frame_tensor)

        return frame_tensor


class DataAugmentation:
    """
    Class for applying data augmentation to video frames.
    """

    @staticmethod
    def random_crop(frame: np.ndarray,
                    crop_size: Tuple[int, int]) -> np.ndarray:
        """Apply random crop to frame"""
        h, w = frame.shape[:2]
        crop_h, crop_w = crop_size

        if h < crop_h or w < crop_w:
            return cv2.resize(frame, crop_size)

        h_start = np.random.randint(0, h - crop_h)
        w_start = np.random.randint(0, w - crop_w)

        return frame[h_start:h_start + crop_h, w_start:w_start + crop_w]

    @staticmethod
    def random_flip(frame: np.ndarray,
                    p: float = 0.5) -> np.ndarray:
        """Apply random horizontal flip"""
        if np.random.random() < p:
            return cv2.flip(frame, 1)
        return frame

    @staticmethod
    def random_rotation(frame: np.ndarray,
                        max_angle: float = 15) -> np.ndarray:
        """Apply random rotation"""
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))

    @staticmethod
    def adjust_brightness(frame: np.ndarray,
                          factor: float) -> np.ndarray:
        """Adjust frame brightness"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class VideoUtils:
    """
    Utility functions for video processing.
    """

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        cap.release()
        return info

    @staticmethod
    def create_video_writer(output_path: str,
                            fps: float,
                            frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """Create a video writer object"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def main():
    """
    Main function for testing the preprocessing functionality.
    """
    # Example usage
    preprocessor = VideoPreprocessor(
        target_size=(64, 64),
        num_frames=16,
        normalize=True,
        augment=True
    )

    # Test video preprocessing
    video_path = "Data/Train/Normal/NormalVideos/video1.mp4"
    if os.path.exists(video_path):
        processed_tensor = preprocessor.preprocess_video(video_path)
        if processed_tensor is not None:
            logging.info(f"Successfully preprocessed video: {video_path}")
            logging.info(f"Output tensor shape: {processed_tensor.shape}")
        else:
            logging.error(f"Failed to preprocess video: {video_path}")


if __name__ == "__main__":
    main()