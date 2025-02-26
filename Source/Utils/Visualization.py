import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
import os
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


class VisualizationManager:
    """
    Class for managing visualizations in the anomaly detection project.

    Created by: nguyenlongCS
    Date: 2025-02-24 11:40:26 UTC
    """

    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize the visualization manager.

        Args:
            save_dir: Directory to save visualization outputs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style for matplotlib
        plt.style.use('seaborn')

        # Define color scheme
        self.colors = {
            'normal': '#2ecc71',
            'anomaly': '#e74c3c',
            'background': '#ecf0f1',
            'grid': '#bdc3c7',
            'text': '#2c3e50'
        }

        logging.info(f"Initialized VisualizationManager with save directory: {save_dir}")

    def plot_training_history(self,
                              history: Dict[str, List[float]],
                              model_name: str) -> None:
        """
        Plot training history metrics.

        Args:
            history: Dictionary containing training metrics
            model_name: Name of the model for saving plots
        """
        try:
            plt.figure(figsize=(12, 5))

            # Create subplot for loss
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Training Loss', color=self.colors['normal'])
            plt.plot(history['val_loss'], label='Validation Loss', color=self.colors['anomaly'])
            plt.title('Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, color=self.colors['grid'])

            # Create subplot for accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Training Accuracy', color=self.colors['normal'])
            plt.plot(history['val_acc'], label='Validation Accuracy', color=self.colors['anomaly'])
            plt.title('Accuracy Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, color=self.colors['grid'])

            # Save plot
            save_path = self.save_dir / f"{model_name}_training_history.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logging.info(f"Training history plot saved to {save_path}")

        except Exception as e:
            logging.error(f"Error plotting training history: {str(e)}")

    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              classes: List[str],
                              model_name: str) -> None:
        """
        Plot confusion matrix for model evaluation.

        Args:
            cm: Confusion matrix array
            classes: List of class names
            model_name: Name of the model
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='YlOrRd',
                xticklabels=classes,
                yticklabels=classes
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            save_path = self.save_dir / f"{model_name}_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logging.info(f"Confusion matrix plot saved to {save_path}")

        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")

    def visualize_attention(self,
                            attention_weights: torch.Tensor,
                            frame: np.ndarray,
                            save_name: str) -> None:
        """
        Visualize attention weights on video frames.

        Args:
            attention_weights: Attention weights from transformer
            frame: Video frame
            save_name: Name for saving the visualization
        """
        try:
            # Reshape attention weights to frame size
            attention_map = attention_weights.cpu().numpy()
            attention_map = cv2.resize(attention_map, (frame.shape[1], frame.shape[0]))

            # Normalize attention map
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            attention_map = np.uint8(255 * attention_map)

            # Apply colormap
            heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

            # Blend original frame with heatmap
            output = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            # Save visualization
            save_path = self.save_dir / f"{save_name}_attention.png"
            cv2.imwrite(str(save_path), output)

            logging.info(f"Attention visualization saved to {save_path}")

        except Exception as e:
            logging.error(f"Error visualizing attention: {str(e)}")

    def plot_anomaly_scores(self,
                            scores: List[float],
                            threshold: float,
                            video_name: str) -> None:
        """
        Plot anomaly scores over time for a video.

        Args:
            scores: List of anomaly scores
            threshold: Anomaly detection threshold
            video_name: Name of the video
        """
        try:
            plt.figure(figsize=(12, 4))

            # Plot scores
            frames = range(len(scores))
            plt.plot(frames, scores, color=self.colors['normal'], label='Anomaly Score')

            # Plot threshold
            plt.axhline(y=threshold, color=self.colors['anomaly'],
                        linestyle='--', label='Threshold')

            # Highlight anomalous regions
            anomalous_frames = [i for i, score in enumerate(scores) if score > threshold]
            if anomalous_frames:
                plt.fill_between(frames, 0, scores,
                                 where=[i in anomalous_frames for i in frames],
                                 color=self.colors['anomaly'], alpha=0.3)

            plt.title(f'Anomaly Scores - {video_name}')
            plt.xlabel('Frame')
            plt.ylabel('Anomaly Score')
            plt.legend()
            plt.grid(True, color=self.colors['grid'])

            save_path = self.save_dir / f"{video_name}_anomaly_scores.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logging.info(f"Anomaly scores plot saved to {save_path}")

        except Exception as e:
            logging.error(f"Error plotting anomaly scores: {str(e)}")

    def create_interactive_dashboard(self,
                                     results: Dict[str, Dict],
                                     save_name: str) -> None:
        """
        Create an interactive dashboard using Plotly.

        Args:
            results: Dictionary containing analysis results
            save_name: Name for saving the dashboard
        """
        try:
            # Create figure with subplots
            fig = go.Figure()

            # Add traces for different metrics
            fig.add_trace(go.Scatter(
                x=list(results['timestamps']),
                y=list(results['anomaly_scores']),
                name='Anomaly Score',
                line=dict(color=self.colors['normal'])
            ))

            # Add threshold line
            fig.add_trace(go.Scatter(
                x=list(results['timestamps']),
                y=[results['threshold']] * len(results['timestamps']),
                name='Threshold',
                line=dict(color=self.colors['anomaly'], dash='dash')
            ))

            # Update layout
            fig.update_layout(
                title='Anomaly Detection Dashboard',
                xaxis_title='Time',
                yaxis_title='Score',
                template='plotly_white',
                hovermode='x unified'
            )

            # Save dashboard
            save_path = self.save_dir / f"{save_name}_dashboard.html"
            fig.write_html(str(save_path))

            logging.info(f"Interactive dashboard saved to {save_path}")

        except Exception as e:
            logging.error(f"Error creating interactive dashboard: {str(e)}")

    def visualize_model_predictions(self,
                                    video_path: str,
                                    predictions: List[Dict],
                                    output_path: str) -> None:
        """
        Create a video with visualized predictions.

        Args:
            video_path: Path to input video
            predictions: List of prediction dictionaries
            output_path: Path to save output video
        """
        try:
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < len(predictions):
                    pred = predictions[frame_idx]

                    # Add prediction visualization
                    label = f"Class: {pred['class']} ({pred['confidence']:.2f})"
                    color = (0, 255, 0) if pred['class'] == 'Normal' else (0, 0, 255)

                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    if pred['is_anomaly']:
                        cv2.rectangle(frame, (0, 0), (width, height),
                                      color, 2)

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()

            logging.info(f"Prediction visualization saved to {output_path}")

        except Exception as e:
            logging.error(f"Error visualizing predictions: {str(e)}")


def main():
    """
    Main function for testing visualization functionality.
    """
    # Create visualization manager
    vis_manager = VisualizationManager()

    # Test training history plot
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.55, 0.45, 0.35],
        'train_acc': [0.8, 0.85, 0.9],
        'val_acc': [0.75, 0.8, 0.85]
    }
    vis_manager.plot_training_history(history, 'test_model')

    # Test confusion matrix plot
    cm = np.array([[50, 10], [5, 35]])
    classes = ['Normal', 'Anomaly']
    vis_manager.plot_confusion_matrix(cm, classes, 'test_model')

    logging.info("Visualization tests completed successfully")


if __name__ == "__main__":
    main()