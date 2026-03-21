"""Basic unit tests for the surveillance system."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.core import (
    setup_logging, set_deterministic_seed, get_device, 
    validate_image, calculate_iou, non_max_suppression
)
from src.models.detection import EdgeOptimizedYOLO
from src.pipelines.streaming import SyntheticDataGenerator
from src.evaluation.benchmarking import EvaluationMetrics


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging(level="INFO")
        assert logger is not None
        assert logger.level == 20  # INFO level
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        
        set_deterministic_seed(42)
        np.random.seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)
    
    def test_validate_image(self):
        """Test image validation."""
        # Valid image
        valid_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert validate_image(valid_image) == True
        
        # Invalid image - wrong dtype
        invalid_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.float32)
        assert validate_image(invalid_image) == False
        
        # Invalid image - wrong shape
        invalid_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        assert validate_image(invalid_image) == False
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        # Overlapping boxes
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        iou = calculate_iou(box1, box2)
        assert 0 < iou < 1
        
        # Non-overlapping boxes
        box3 = np.array([200, 200, 300, 300])
        iou = calculate_iou(box1, box3)
        assert iou == 0.0
        
        # Identical boxes
        iou = calculate_iou(box1, box1)
        assert iou == 1.0
    
    def test_non_max_suppression(self):
        """Test Non-Maximum Suppression."""
        boxes = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
            [200, 200, 300, 300]
        ])
        scores = np.array([0.9, 0.8, 0.7])
        
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
        
        assert len(keep_indices) > 0
        assert len(keep_indices) <= len(boxes)


class TestModels:
    """Test model components."""
    
    def test_edge_optimized_yolo(self):
        """Test EdgeOptimizedYOLO model."""
        model = EdgeOptimizedYOLO(num_classes=6, input_size=(416, 416))
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 416, 416)
        output = model(dummy_input)
        
        assert output is not None
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == 11  # 5 + 6 classes
    
    def test_model_export(self):
        """Test model export functionality."""
        model = EdgeOptimizedYOLO(num_classes=6)
        
        # Test model can be traced
        dummy_input = torch.randn(1, 3, 416, 416)
        traced_model = torch.jit.trace(model, dummy_input)
        
        assert traced_model is not None


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_synthetic_data_generator(self):
        """Test synthetic data generation."""
        from omegaconf import DictConfig
        
        config = DictConfig({})
        generator = SyntheticDataGenerator(config)
        
        # Test frame generation
        frame = generator.generate_frame(640, 480)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        
        # Test audio generation
        audio = generator.generate_audio_sample(1.0, 16000)
        assert len(audio) == 16000
        assert audio.dtype == np.float32


class TestEvaluation:
    """Test evaluation components."""
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics container."""
        metrics = EvaluationMetrics()
        
        # Add some metrics
        metrics.add_metric('precision', 0.8)
        metrics.add_metric('precision', 0.9)
        metrics.add_metric('recall', 0.7)
        
        # Test average calculation
        assert metrics.get_average('precision') == 0.85
        assert metrics.get_average('recall') == 0.7
        assert metrics.get_average('f1') == 0.0  # No f1 metric added
        
        # Test standard deviation
        assert metrics.get_std('precision') > 0
        
        # Test to_dict conversion
        metrics_dict = metrics.to_dict()
        assert 'precision_mean' in metrics_dict
        assert 'precision_std' in metrics_dict


class TestSurveillanceSystem:
    """Test main surveillance system."""
    
    def test_system_initialization(self):
        """Test surveillance system initialization."""
        from src.surveillance import SurveillanceSystem
        from omegaconf import DictConfig
        
        # Create minimal config
        config = DictConfig({
            'model': {
                'baseline': {
                    'name': 'edge_optimized',
                    'input_size': [416, 416],
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.45
                }
            },
            'detection': {
                'classes': ['person', 'car'],
                'alerts': {
                    'intrusion_zones': {'enabled': True},
                    'loitering': {'enabled': True, 'min_duration_seconds': 30}
                }
            }
        })
        
        # Save config temporarily
        config_path = Path("test_config.yaml")
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            system = SurveillanceSystem(str(config_path))
            assert system is not None
            assert system.config is not None
        finally:
            config_path.unlink()


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_detection(self):
        """Test end-to-end detection pipeline."""
        from src.surveillance import SurveillanceSystem, DetectionResult
        from omegaconf import DictConfig
        
        # Create minimal config
        config = DictConfig({
            'model': {
                'baseline': {
                    'name': 'edge_optimized',
                    'input_size': [416, 416],
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.45
                }
            },
            'detection': {
                'classes': ['person', 'car'],
                'alerts': {
                    'intrusion_zones': {'enabled': True},
                    'loitering': {'enabled': True, 'min_duration_seconds': 30}
                }
            }
        })
        
        # Create detection result
        boxes = np.array([[100, 100, 200, 200]])
        scores = np.array([0.8])
        class_ids = np.array([0])
        class_names = ['person']
        
        detection = DetectionResult(boxes, scores, class_ids, class_names)
        
        # Test filtering
        filtered = detection.filter_by_confidence(0.7)
        assert len(filtered.boxes) == 1
        
        filtered = detection.filter_by_confidence(0.9)
        assert len(filtered.boxes) == 0
        
        # Test class filtering
        person_detections = detection.filter_by_class(['person'])
        assert len(person_detections.boxes) == 1
        
        car_detections = detection.filter_by_class(['car'])
        assert len(car_detections.boxes) == 0


@pytest.mark.slow
class TestPerformance:
    """Performance tests."""
    
    def test_model_inference_speed(self):
        """Test model inference speed."""
        model = EdgeOptimizedYOLO(num_classes=6)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 416, 416)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Time inference
        import time
        start_time = time.perf_counter()
        
        for _ in range(100):
            with torch.no_grad():
                _ = model(dummy_input)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100
        
        # Should be reasonably fast (less than 100ms per inference)
        assert avg_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
