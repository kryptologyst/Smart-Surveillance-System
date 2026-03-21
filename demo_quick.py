#!/usr/bin/env python3
"""
Quick demo script for Smart Surveillance System.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.core import setup_logging, set_deterministic_seed
from src.surveillance import SurveillanceSystem
from src.pipelines.streaming import SyntheticDataGenerator
from src.models.detection import EdgeOptimizedYOLO


async def run_demo():
    """Run a quick demo of the surveillance system."""
    print("🔍 Smart Surveillance System Demo")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Set deterministic seed
    set_deterministic_seed(42)
    
    print("✅ Initialized logging and random seed")
    
    # Test model creation
    print("\n🤖 Testing model creation...")
    model = EdgeOptimizedYOLO(num_classes=6, input_size=(416, 416))
    print(f"✅ Created EdgeOptimizedYOLO model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test synthetic data generation
    print("\n📊 Testing synthetic data generation...")
    from omegaconf import DictConfig
    config = DictConfig({})
    generator = SyntheticDataGenerator(config)
    
    frame = generator.generate_frame(640, 480)
    print(f"✅ Generated synthetic frame: {frame.shape}")
    
    audio = generator.generate_audio_sample(1.0, 16000)
    print(f"✅ Generated synthetic audio: {len(audio)} samples")
    
    # Test surveillance system initialization
    print("\n🎯 Testing surveillance system...")
    try:
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
                'classes': ['person', 'car', 'bicycle'],
                'alerts': {
                    'intrusion_zones': {'enabled': True},
                    'loitering': {'enabled': True, 'min_duration_seconds': 30}
                }
            },
            'streaming': {
                'camera': {'source': None, 'resolution': [640, 480], 'fps': 30, 'buffer_size': 5}
            },
            'mqtt': {'enabled': False},
            'websocket': {'enabled': False}
        })
        
        # Save config temporarily
        import yaml
        config_path = Path("demo_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        system = SurveillanceSystem(str(config_path))
        print("✅ Surveillance system initialized")
        
        # Test detection on synthetic frame
        detections = system.detect_objects(frame)
        print(f"✅ Detection completed: {detections.num_detections} detections")
        
        # Test alert analysis
        alerts = system.analyze_detections(detections)
        print(f"✅ Alert analysis completed: {len(alerts)} alerts")
        
        # Cleanup
        config_path.unlink()
        
    except Exception as e:
        print(f"⚠️ Surveillance system test failed: {e}")
    
    # Test evaluation metrics
    print("\n📈 Testing evaluation metrics...")
    from src.evaluation.benchmarking import EvaluationMetrics
    
    metrics = EvaluationMetrics()
    metrics.add_metric('precision', 0.85)
    metrics.add_metric('recall', 0.78)
    metrics.add_metric('f1', 0.81)
    
    print(f"✅ Evaluation metrics: Precision={metrics.get_average('precision'):.2f}, "
          f"Recall={metrics.get_average('recall'):.2f}, F1={metrics.get_average('f1'):.2f}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the full system:")
    print("1. streamlit run demo/app.py  # Interactive web demo")
    print("2. python main.py --demo     # Command-line demo")
    print("3. python scripts/train.py --action train  # Train models")


def main():
    """Main demo function."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
