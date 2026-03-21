"""Core utilities for the Smart Surveillance System."""

import random
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import cv2
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up structured logging for the surveillance system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("surveillance")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Preferred device ('cuda', 'cpu', 'mps')
        
    Returns:
        Available torch device
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
    """
    OmegaConf.save(config, output_path)


def create_output_dirs(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """Create necessary output directories.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Dictionary mapping directory names to paths
    """
    base_path = Path(base_dir)
    dirs = {
        "models": base_path / "models",
        "logs": base_path / "logs",
        "assets": base_path / "assets",
        "data_processed": base_path / "data" / "processed",
        "exports": base_path / "exports",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """Initialize timer.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Optional logger for output
        """
        self.operation_name = operation_name
        self.logger = logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> 'PerformanceTimer':
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and log result."""
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        
        if self.logger:
            self.logger.info(f"{self.operation_name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not completed")
        return self.end_time - self.start_time


def validate_image(image: np.ndarray) -> bool:
    """Validate image array for processing.
    
    Args:
        image: Image array to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    
    if image.dtype != np.uint8:
        return False
    
    return True


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> Tuple[np.ndarray, float]:
    """Resize image while optionally maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        return padded, scale
    else:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized, 1.0


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                       iou_threshold: float = 0.5) -> List[int]:
    """Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by scores in descending order
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        # Pick the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_indices = indices[1:]
        remaining_boxes = boxes[remaining_indices]
        
        ious = np.array([
            calculate_iou(current_box, box) for box in remaining_boxes
        ])
        
        # Keep boxes with IoU below threshold
        indices = remaining_indices[ious <= iou_threshold]
    
    return keep
