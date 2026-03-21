"""Main surveillance system implementation."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from .utils.core import (
    PerformanceTimer, get_device, load_config, setup_logging, 
    set_deterministic_seed, validate_image, calculate_iou, non_max_suppression
)
from .models.detection import EdgeOptimizedYOLO, QuantizedModel, load_pretrained_model
from .pipelines.streaming import DataPipeline, SyntheticDataGenerator


logger = logging.getLogger("surveillance.system")


class DetectionResult:
    """Container for detection results."""
    
    def __init__(self, boxes: np.ndarray, scores: np.ndarray, 
                 class_ids: np.ndarray, class_names: List[str]):
        """Initialize detection result.
        
        Args:
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            class_names: List of class names
        """
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.timestamp = time.time()
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """Filter detections by confidence threshold.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Filtered detection result
        """
        mask = self.scores >= threshold
        return DetectionResult(
            self.boxes[mask],
            self.scores[mask],
            self.class_ids[mask],
            self.class_names
        )
    
    def filter_by_class(self, target_classes: List[str]) -> 'DetectionResult':
        """Filter detections by target classes.
        
        Args:
            target_classes: List of target class names
            
        Returns:
            Filtered detection result
        """
        target_ids = [i for i, name in enumerate(self.class_names) 
                     if name in target_classes]
        
        if not target_ids:
            return DetectionResult(
                np.array([]), np.array([]), np.array([]), self.class_names
            )
        
        mask = np.isin(self.class_ids, target_ids)
        return DetectionResult(
            self.boxes[mask],
            self.scores[mask],
            self.class_ids[mask],
            self.class_names
        )
    
    def apply_nms(self, iou_threshold: float = 0.5) -> 'DetectionResult':
        """Apply Non-Maximum Suppression.
        
        Args:
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Detection result after NMS
        """
        if len(self.boxes) == 0:
            return self
        
        keep_indices = non_max_suppression(self.boxes, self.scores, iou_threshold)
        
        return DetectionResult(
            self.boxes[keep_indices],
            self.scores[keep_indices],
            self.class_ids[keep_indices],
            self.class_names
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'boxes': self.boxes.tolist(),
            'scores': self.scores.tolist(),
            'class_ids': self.class_ids.tolist(),
            'class_names': [self.class_names[i] for i in self.class_ids],
            'timestamp': self.timestamp,
            'num_detections': len(self.boxes)
        }


class SurveillanceSystem:
    """Main surveillance system class."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize surveillance system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config.get('log_level', 'INFO'),
            log_file=self.config.get('log_file')
        )
        
        # Set deterministic seed
        set_deterministic_seed(self.config.get('seed', 42))
        
        # Initialize components
        self.device = get_device()
        self.model = None
        self.data_pipeline = None
        self.synthetic_generator = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'avg_fps': 0.0,
            'start_time': time.time()
        }
        
        # Alert tracking
        self.alert_history = []
        self.intrusion_zones = []
        self.loitering_tracker = {}  # Track objects for loitering detection
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize system components."""
        try:
            # Initialize model
            self._load_model()
            
            # Initialize data pipeline
            self.data_pipeline = DataPipeline(self.config)
            
            # Initialize synthetic data generator for testing
            self.synthetic_generator = SyntheticDataGenerator(self.config)
            
            self.logger.info("Surveillance system components initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _load_model(self) -> None:
        """Load detection model."""
        model_config = self.config.model.baseline
        
        try:
            # Try to load pretrained model first
            self.model = load_pretrained_model(
                model_config.name, 
                len(self.config.detection.classes)
            )
            
            # Move to device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            self.logger.info(f"Model loaded: {model_config.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained model: {e}")
            # Fallback to custom model
            self.model = EdgeOptimizedYOLO(
                len(self.config.detection.classes),
                tuple(model_config.input_size)
            ).to(self.device)
            
            self.logger.info("Using custom edge-optimized model")
    
    async def start(self) -> bool:
        """Start surveillance system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Start data pipeline
            if not await self.data_pipeline.start():
                self.logger.error("Failed to start data pipeline")
                return False
            
            self.logger.info("Surveillance system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting surveillance system: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop surveillance system."""
        if self.data_pipeline:
            await self.data_pipeline.stop()
        
        self.logger.info("Surveillance system stopped")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        if not validate_image(frame):
            raise ValueError("Invalid input frame")
        
        # Resize to model input size
        target_size = tuple(self.config.model.baseline.input_size)
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor_frame = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0)
        
        return tensor_frame.to(self.device)
    
    def postprocess_detections(self, model_output: torch.Tensor, 
                              original_shape: Tuple[int, int]) -> DetectionResult:
        """Postprocess model output to get detections.
        
        Args:
            model_output: Raw model output
            original_shape: Original frame shape (height, width)
            
        Returns:
            Detection result
        """
        # Convert to numpy
        output = model_output.cpu().numpy()
        
        # Extract detections (simplified - assumes YOLO format)
        # In practice, this would depend on the specific model architecture
        boxes = []
        scores = []
        class_ids = []
        
        # This is a simplified postprocessing - real implementation would
        # depend on the specific model format
        if len(output.shape) == 4:  # Batch format
            output = output[0]  # Remove batch dimension
        
        # Placeholder implementation - would need actual YOLO postprocessing
        # For now, return empty detections
        boxes = np.array([])
        scores = np.array([])
        class_ids = np.array([])
        
        return DetectionResult(
            boxes, scores, class_ids, self.config.detection.classes
        )
    
    def detect_objects(self, frame: np.ndarray) -> DetectionResult:
        """Detect objects in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Detection result
        """
        with PerformanceTimer("Object detection", self.logger):
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'predict'):
                    # For Ultralytics YOLO models
                    results = self.model.predict(input_tensor)
                    # Convert results to our format
                    boxes = np.array([])
                    scores = np.array([])
                    class_ids = np.array([])
                else:
                    # For custom models
                    output = self.model(input_tensor)
                    # Postprocess
                    detection_result = self.postprocess_detections(output, frame.shape[:2])
                    return detection_result
        
        # For now, return empty detections
        return DetectionResult(
            np.array([]), np.array([]), np.array([]), 
            self.config.detection.classes
        )
    
    def analyze_detections(self, detections: DetectionResult) -> List[Dict[str, Any]]:
        """Analyze detections for surveillance alerts.
        
        Args:
            detections: Detection results
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Filter for person detections
        person_detections = detections.filter_by_class(['person'])
        
        if len(person_detections.boxes) > 0:
            # Check for intrusion zones
            if self.config.detection.alerts.intrusion_zones.enabled:
                intrusion_alerts = self._check_intrusion_zones(person_detections)
                alerts.extend(intrusion_alerts)
            
            # Check for loitering
            if self.config.detection.alerts.loitering.enabled:
                loitering_alerts = self._check_loitering(person_detections)
                alerts.extend(loitering_alerts)
        
        return alerts
    
    def _check_intrusion_zones(self, detections: DetectionResult) -> List[Dict[str, Any]]:
        """Check for intrusions in defined zones.
        
        Args:
            detections: Person detections
            
        Returns:
            List of intrusion alerts
        """
        alerts = []
        
        for i, box in enumerate(detections.boxes):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            # Check if person is in any intrusion zone
            for zone in self.intrusion_zones:
                if self._point_in_polygon((center_x, center_y), zone['polygon']):
                    alert = {
                        'type': 'intrusion',
                        'zone_id': zone['id'],
                        'zone_name': zone['name'],
                        'person_id': i,
                        'confidence': detections.scores[i],
                        'timestamp': time.time(),
                        'location': {'x': center_x, 'y': center_y}
                    }
                    alerts.append(alert)
        
        return alerts
    
    def _check_loitering(self, detections: DetectionResult) -> List[Dict[str, Any]]:
        """Check for loitering behavior.
        
        Args:
            detections: Person detections
            
        Returns:
            List of loitering alerts
        """
        alerts = []
        current_time = time.time()
        min_duration = self.config.detection.alerts.loitering.min_duration_seconds
        
        for i, box in enumerate(detections.boxes):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            # Simple loitering detection based on position persistence
            person_key = f"person_{i}"
            
            if person_key not in self.loitering_tracker:
                self.loitering_tracker[person_key] = {
                    'first_seen': current_time,
                    'last_position': (center_x, center_y),
                    'positions': [(center_x, center_y, current_time)]
                }
            else:
                tracker = self.loitering_tracker[person_key]
                
                # Check if person has moved significantly
                last_pos = tracker['last_position']
                distance = np.sqrt((center_x - last_pos[0])**2 + (center_y - last_pos[1])**2)
                
                if distance < 50:  # Person hasn't moved much
                    duration = current_time - tracker['first_seen']
                    
                    if duration >= min_duration:
                        alert = {
                            'type': 'loitering',
                            'person_id': i,
                            'confidence': detections.scores[i],
                            'duration': duration,
                            'timestamp': current_time,
                            'location': {'x': center_x, 'y': center_y}
                        }
                        alerts.append(alert)
                else:
                    # Reset tracking
                    self.loitering_tracker[person_key] = {
                        'first_seen': current_time,
                        'last_position': (center_x, center_y),
                        'positions': [(center_x, center_y, current_time)]
                    }
        
        return alerts
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon.
        
        Args:
            point: Point coordinates (x, y)
            polygon: List of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def draw_detections(self, frame: np.ndarray, detections: DetectionResult) -> np.ndarray:
        """Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: Detection results
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for i, (box, score, class_id) in enumerate(zip(
            detections.boxes, detections.scores, detections.class_ids
        )):
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            class_name = detections.class_names[class_id]
            label = f"{class_name}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background rectangle
            cv2.rectangle(
                result_frame, (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), (0, 255, 0), -1
            )
            
            # Draw text
            cv2.putText(
                result_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )
        
        return result_frame
    
    def update_performance_metrics(self, latency_ms: float) -> None:
        """Update performance metrics.
        
        Args:
            latency_ms: Inference latency in milliseconds
        """
        self.performance_metrics['total_frames'] += 1
        
        # Update average latency
        current_avg = self.performance_metrics['avg_latency_ms']
        total_frames = self.performance_metrics['total_frames']
        
        self.performance_metrics['avg_latency_ms'] = (
            (current_avg * (total_frames - 1) + latency_ms) / total_frames
        )
        
        # Update FPS
        elapsed_time = time.time() - self.performance_metrics['start_time']
        self.performance_metrics['avg_fps'] = total_frames / elapsed_time
    
    async def run_surveillance_loop(self) -> None:
        """Main surveillance loop."""
        self.logger.info("Starting surveillance loop")
        
        while True:
            try:
                # Get frame from pipeline
                frame = self.data_pipeline.get_frame()
                
                if frame is None:
                    # Use synthetic data if no real frame available
                    frame = self.synthetic_generator.generate_frame()
                
                # Detect objects
                start_time = time.perf_counter()
                detections = self.detect_objects(frame)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                self.update_performance_metrics(latency_ms)
                
                # Analyze detections for alerts
                alerts = self.analyze_detections(detections)
                
                # Publish detections and alerts
                if detections.num_detections > 0:
                    detection_data = detections.to_dict()
                    self.data_pipeline.publish_detection(detection_data)
                
                if alerts:
                    for alert in alerts:
                        self.data_pipeline.publish_alert(alert)
                        self.alert_history.append(alert)
                
                # Draw detections on frame
                result_frame = self.draw_detections(frame, detections)
                
                # Broadcast frame to WebSocket clients
                await self.data_pipeline.broadcast_frame(result_frame)
                
                # Log performance every 100 frames
                if self.performance_metrics['total_frames'] % 100 == 0:
                    self.logger.info(
                        f"Processed {self.performance_metrics['total_frames']} frames, "
                        f"Avg FPS: {self.performance_metrics['avg_fps']:.1f}, "
                        f"Avg Latency: {self.performance_metrics['avg_latency_ms']:.1f}ms"
                    )
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in surveillance loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
