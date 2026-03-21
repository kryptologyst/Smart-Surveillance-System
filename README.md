# Smart Surveillance System

Edge-optimized surveillance system using computer vision and AI for real-time object detection, intrusion monitoring, and loitering detection.

## DISCLAIMER

**This is a research/educational demonstration project. NOT FOR SAFETY-CRITICAL DEPLOYMENT. Use at your own risk.**

## Features

- **Real-time Object Detection**: Person detection using edge-optimized YOLO models
- **Intrusion Detection**: Configurable intrusion zones with real-time alerts
- **Loitering Detection**: Time-based loitering behavior analysis
- **Edge Optimization**: Quantization, pruning, and hardware-specific optimizations
- **Multi-format Export**: ONNX, TensorFlow Lite, CoreML, OpenVINO support
- **Live Streaming**: WebSocket and MQTT integration for real-time data
- **Performance Monitoring**: Comprehensive metrics and benchmarking
- **Interactive Demo**: Streamlit-based web interface

## Architecture

```
src/
├── models/           # Model definitions and optimization
├── pipelines/        # Data streaming and I/O
├── evaluation/       # Benchmarking and evaluation tools
├── export/          # Model export and deployment
├── utils/           # Core utilities and helpers
└── surveillance.py  # Main surveillance system

configs/             # Configuration files
├── device/         # Device-specific configs
├── quant/          # Quantization settings
└── comms/          # Communication configs

demo/               # Interactive demo application
scripts/            # Training and deployment scripts
tests/              # Unit tests
assets/             # Outputs and visualizations
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- OpenCV 4.8+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Smart-Surveillance-System.git
cd Smart-Surveillance-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the demo:
```bash
streamlit run demo/app.py
```

### Basic Usage

1. **Start the surveillance system**:
```bash
python main.py --config configs/device/surveillance.yaml
```

2. **Run with demo data**:
```bash
python main.py --demo --config configs/device/surveillance.yaml
```

3. **Train a custom model**:
```bash
python scripts/train.py --action train --config configs/device/surveillance.yaml
```

4. **Optimize model for edge deployment**:
```bash
python scripts/train.py --action optimize --model-path outputs/trained_model.pth
```

## Configuration

### Device Configurations

The system supports multiple edge device targets:

- **Raspberry Pi 4B**: Optimized for ARM CPU with TensorFlow Lite
- **NVIDIA Jetson Nano**: CUDA acceleration with ONNX Runtime
- **NVIDIA Jetson Xavier**: High-performance inference
- **Android Devices**: Mobile-optimized TensorFlow Lite models

### Model Configurations

- **Baseline Model**: Higher accuracy YOLOv8n
- **Edge Optimized**: Quantized and pruned for low latency
- **Custom Tiny**: Ultra-lightweight custom architecture

### Detection Classes

Default detection classes:
- person
- car
- bicycle
- motorcycle
- bus
- truck

## Performance Metrics

### Accuracy Metrics
- **mAP**: Mean Average Precision
- **Precision**: True positive rate
- **Recall**: Detection rate
- **F1 Score**: Harmonic mean of precision and recall

### Efficiency Metrics
- **Latency**: Inference time (p50, p95, p99)
- **Throughput**: Frames per second
- **Memory Usage**: Peak RAM consumption
- **Model Size**: Compressed model size
- **Energy Consumption**: Power usage per inference

### Sample Performance (Raspberry Pi 4B)
- **YOLOv8n Baseline**: 25 FPS, 40ms latency, 800MB RAM
- **YOLOv8n Quantized**: 35 FPS, 28ms latency, 600MB RAM
- **Edge Optimized**: 45 FPS, 22ms latency, 400MB RAM

## Deployment

### Edge Device Setup

1. **Raspberry Pi Setup**:
```bash
python scripts/setup_device.py --device raspberry_pi
```

2. **Jetson Setup**:
```bash
python scripts/setup_device.py --device jetson_nano
```

### Model Export

Export models to various edge formats:

```python
from src.export.deployment import ModelDeployment

deployment = ModelDeployment(config)
exported_models = deployment.export_to_edge_formats(model, "exports/")
```

### Deployment Package

Create deployment packages for specific devices:

```python
package_path = deployment.create_deployment_package(exported_models, "raspberry_pi")
```

## API Reference

### SurveillanceSystem

Main surveillance system class.

```python
from src.surveillance import SurveillanceSystem

system = SurveillanceSystem("configs/device/surveillance.yaml")
await system.start()
await system.run_surveillance_loop()
```

### DetectionResult

Container for detection results.

```python
detections = system.detect_objects(frame)
filtered = detections.filter_by_confidence(0.5)
alerts = system.analyze_detections(filtered)
```

### DataPipeline

Streaming data pipeline.

```python
from src.pipelines.streaming import DataPipeline

pipeline = DataPipeline(config)
await pipeline.start()
frame = pipeline.get_frame()
```

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Google-style** docstrings

### Running Tests

```bash
pytest tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Limitations

- **Accuracy**: Edge-optimized models may have reduced accuracy
- **Real-time Constraints**: Performance depends on hardware capabilities
- **Privacy**: No built-in privacy protection (faces/license plates)
- **Security**: Basic authentication only
- **Scalability**: Single-device deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational/research purposes. See LICENSE file for details.

## Acknowledgments

- Ultralytics YOLO for baseline models
- OpenCV for computer vision utilities
- PyTorch for deep learning framework
- Streamlit for demo interface

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

---

**Remember**: This is a demonstration project. Not suitable for production surveillance systems without proper security, privacy, and safety measures.
# Smart-Surveillance-System
