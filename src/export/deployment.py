"""Model export and deployment utilities."""

import logging
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
import numpy as np
from omegaconf import DictConfig

from ..utils.core import PerformanceTimer, get_device, logger
from ..models.detection import EdgeOptimizedYOLO, ModelExporter


class ModelDeployment:
    """Model deployment utilities for edge devices."""
    
    def __init__(self, config: DictConfig):
        """Initialize model deployment.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.device = get_device()
        self.exporter = ModelExporter(config)
        
    def export_to_edge_formats(self, model: torch.nn.Module, 
                              output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Export model to various edge formats.
        
        Args:
            model: PyTorch model to export
            output_dir: Output directory
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_models = {}
        
        # Export to ONNX
        if self.config.hardware.openvino.enabled:
            onnx_path = output_dir / "model.onnx"
            self.exporter.export_to_onnx(model, onnx_path)
            exported_models['onnx'] = onnx_path
            
            # Convert ONNX to OpenVINO
            if self._check_openvino_available():
                openvino_path = output_dir / "openvino_model"
                self._convert_to_openvino(onnx_path, openvino_path)
                exported_models['openvino'] = openvino_path
        
        # Export to TensorFlow Lite
        if self.config.hardware.tensorrt.enabled or self.config.quantization.ptq.enabled:
            tflite_path = output_dir / "model.tflite"
            if 'onnx' in exported_models:
                self.exporter.export_to_tflite(exported_models['onnx'], tflite_path)
                exported_models['tflite'] = tflite_path
        
        # Export to CoreML
        if self.config.hardware.coreml.enabled:
            coreml_path = output_dir / "model.mlmodel"
            self._export_to_coreml(model, coreml_path)
            exported_models['coreml'] = coreml_path
        
        logger.info(f"Exported models to: {output_dir}")
        return exported_models
    
    def _check_openvino_available(self) -> bool:
        """Check if OpenVINO is available."""
        try:
            import openvino
            return True
        except ImportError:
            logger.warning("OpenVINO not available")
            return False
    
    def _convert_to_openvino(self, onnx_path: Path, output_path: Path) -> None:
        """Convert ONNX model to OpenVINO format.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output directory for OpenVINO model
        """
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
            
            # Convert ONNX to OpenVINO IR
            ov_model = mo.convert_model(str(onnx_path))
            
            # Serialize model
            serialize(ov_model, str(output_path / "model.xml"))
            
            logger.info(f"OpenVINO model saved to: {output_path}")
            
        except ImportError:
            logger.error("OpenVINO tools not available")
            raise
    
    def _export_to_coreml(self, model: torch.nn.Module, output_path: Path) -> None:
        """Export PyTorch model to CoreML format.
        
        Args:
            model: PyTorch model
            output_path: Output CoreML model path
        """
        try:
            import coremltools as ct
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 416, 416)
            
            # Trace model
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=dummy_input.shape)]
            )
            
            # Save model
            coreml_model.save(str(output_path))
            
            logger.info(f"CoreML model saved to: {output_path}")
            
        except ImportError:
            logger.error("CoreML tools not available")
            raise
    
    def create_deployment_package(self, model_paths: Dict[str, Path], 
                                 target_device: str) -> Path:
        """Create deployment package for target device.
        
        Args:
            model_paths: Dictionary of model format to path
            target_device: Target device type
            
        Returns:
            Path to deployment package
        """
        package_dir = Path("deployments") / target_device
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy models
        models_dir = package_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for format_name, model_path in model_paths.items():
            if model_path.exists():
                if model_path.is_dir():
                    # Copy directory
                    import shutil
                    shutil.copytree(model_path, models_dir / format_name)
                else:
                    # Copy file
                    import shutil
                    shutil.copy2(model_path, models_dir / f"{format_name}.{model_path.suffix[1:]}")
        
        # Create device-specific configuration
        device_config = self._create_device_config(target_device)
        config_path = package_dir / "config.yaml"
        self._save_config(device_config, config_path)
        
        # Create deployment script
        script_path = package_dir / "deploy.py"
        self._create_deployment_script(script_path, target_device)
        
        # Create requirements file
        requirements_path = package_dir / "requirements.txt"
        self._create_requirements_file(requirements_path, target_device)
        
        logger.info(f"Deployment package created: {package_dir}")
        return package_dir
    
    def _create_device_config(self, target_device: str) -> Dict[str, Any]:
        """Create device-specific configuration.
        
        Args:
            target_device: Target device type
            
        Returns:
            Device configuration
        """
        device_configs = {
            'raspberry_pi': {
                'model_format': 'tflite',
                'input_size': [416, 416],
                'batch_size': 1,
                'num_threads': 4,
                'memory_limit_mb': 1000
            },
            'jetson_nano': {
                'model_format': 'onnx',
                'input_size': [640, 640],
                'batch_size': 1,
                'num_threads': 4,
                'memory_limit_mb': 2000
            },
            'jetson_xavier': {
                'model_format': 'onnx',
                'input_size': [640, 640],
                'batch_size': 1,
                'num_threads': 6,
                'memory_limit_mb': 4000
            },
            'android': {
                'model_format': 'tflite',
                'input_size': [416, 416],
                'batch_size': 1,
                'num_threads': 8,
                'memory_limit_mb': 1500
            }
        }
        
        return device_configs.get(target_device, device_configs['raspberry_pi'])
    
    def _save_config(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        import yaml
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _create_deployment_script(self, script_path: Path, target_device: str) -> None:
        """Create deployment script for target device.
        
        Args:
            script_path: Script output path
            target_device: Target device type
        """
        script_content = f'''#!/usr/bin/env python3
"""
Deployment script for {target_device}.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from surveillance import SurveillanceSystem

def main():
    """Main deployment function."""
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    
    # Initialize surveillance system
    system = SurveillanceSystem(config_path)
    
    # Start system
    import asyncio
    asyncio.run(system.start())
    
    # Run surveillance loop
    asyncio.run(system.run_surveillance_loop())

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
    
    def _create_requirements_file(self, requirements_path: Path, target_device: str) -> None:
        """Create requirements file for target device.
        
        Args:
            requirements_path: Requirements file path
            target_device: Target device type
        """
        base_requirements = [
            "numpy>=1.24.0",
            "opencv-python>=4.8.0",
            "paho-mqtt>=1.6.0",
            "websockets>=11.0.0",
            "pyyaml>=6.0",
            "omegaconf>=2.3.0",
        ]
        
        device_specific = {
            'raspberry_pi': [
                "tflite-runtime>=2.13.0",
            ],
            'jetson_nano': [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "onnxruntime>=1.15.0",
            ],
            'jetson_xavier': [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "onnxruntime>=1.15.0",
            ],
            'android': [
                "tflite-runtime>=2.13.0",
            ]
        }
        
        requirements = base_requirements + device_specific.get(target_device, [])
        
        with open(requirements_path, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")


class EdgeDeviceSetup:
    """Setup utilities for edge devices."""
    
    def __init__(self, config: DictConfig):
        """Initialize edge device setup.
        
        Args:
            config: Setup configuration
        """
        self.config = config
    
    def setup_raspberry_pi(self) -> None:
        """Setup Raspberry Pi environment."""
        logger.info("Setting up Raspberry Pi environment")
        
        # Install system dependencies
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pip python3-venv",
            "sudo apt-get install -y libopencv-dev python3-opencv",
            "sudo apt-get install -y libatlas-base-dev",
            "sudo apt-get install -y libhdf5-dev libhdf5-serial-dev",
        ]
        
        for cmd in commands:
            self._run_command(cmd)
        
        # Setup Python environment
        self._setup_python_env()
        
        logger.info("Raspberry Pi setup completed")
    
    def setup_jetson(self) -> None:
        """Setup NVIDIA Jetson environment."""
        logger.info("Setting up NVIDIA Jetson environment")
        
        # Install system dependencies
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pip python3-venv",
            "sudo apt-get install -y libopencv-dev python3-opencv",
            "sudo apt-get install -y libcudnn8 libcudnn8-dev",
        ]
        
        for cmd in commands:
            self._run_command(cmd)
        
        # Setup Python environment
        self._setup_python_env()
        
        # Install PyTorch for Jetson
        self._install_jetson_pytorch()
        
        logger.info("NVIDIA Jetson setup completed")
    
    def setup_android(self) -> None:
        """Setup Android environment."""
        logger.info("Setting up Android environment")
        
        # This would typically involve Android Studio setup
        # and creating an Android project with TensorFlow Lite
        
        logger.info("Android setup instructions:")
        logger.info("1. Install Android Studio")
        logger.info("2. Create new Android project")
        logger.info("3. Add TensorFlow Lite dependency")
        logger.info("4. Copy model files to assets/")
        logger.info("5. Implement inference code")
    
    def _run_command(self, command: str) -> None:
        """Run system command.
        
        Args:
            command: Command to run
        """
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            logger.info(f"Command executed: {command}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {e.stderr}")
            raise
    
    def _setup_python_env(self) -> None:
        """Setup Python virtual environment."""
        commands = [
            "python3 -m venv venv",
            "source venv/bin/activate",
            "pip install --upgrade pip",
        ]
        
        for cmd in commands:
            self._run_command(cmd)
    
    def _install_jetson_pytorch(self) -> None:
        """Install PyTorch for Jetson."""
        # This would install the appropriate PyTorch wheel for Jetson
        logger.info("Installing PyTorch for Jetson...")
        
        # Example command (would need actual Jetson-specific wheel)
        # self._run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


class OTAManager:
    """Over-the-Air update manager."""
    
    def __init__(self, config: DictConfig):
        """Initialize OTA manager.
        
        Args:
            config: OTA configuration
        """
        self.config = config
        self.current_version = "1.0.0"
        self.update_server = config.get('update_server', 'http://localhost:8000')
    
    def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """Check for available updates.
        
        Returns:
            Update information or None if no updates available
        """
        try:
            import requests
            
            response = requests.get(f"{self.update_server}/api/updates/check")
            response.raise_for_status()
            
            update_info = response.json()
            
            if update_info['version'] != self.current_version:
                return update_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return None
    
    def download_update(self, update_info: Dict[str, Any]) -> Path:
        """Download update package.
        
        Args:
            update_info: Update information
            
        Returns:
            Path to downloaded update package
        """
        try:
            import requests
            import zipfile
            
            # Download update package
            response = requests.get(update_info['download_url'])
            response.raise_for_status()
            
            # Save to temporary file
            update_path = Path(f"update_{update_info['version']}.zip")
            with open(update_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Update downloaded: {update_path}")
            return update_path
            
        except Exception as e:
            logger.error(f"Error downloading update: {e}")
            raise
    
    def install_update(self, update_path: Path) -> bool:
        """Install update package.
        
        Args:
            update_path: Path to update package
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import zipfile
            import shutil
            
            # Extract update package
            extract_dir = Path("update_extract")
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(update_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Backup current installation
            backup_dir = Path(f"backup_{self.current_version}")
            if Path("src").exists():
                shutil.copytree("src", backup_dir / "src")
            
            # Install new version
            if (extract_dir / "src").exists():
                shutil.copytree(extract_dir / "src", "src", dirs_exist_ok=True)
            
            # Update version
            self.current_version = "1.1.0"  # Would be read from update package
            
            # Cleanup
            shutil.rmtree(extract_dir)
            update_path.unlink()
            
            logger.info("Update installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing update: {e}")
            return False
    
    def rollback_update(self) -> bool:
        """Rollback to previous version.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            
            # Find latest backup
            backup_dirs = list(Path(".").glob("backup_*"))
            if not backup_dirs:
                logger.error("No backup found for rollback")
                return False
            
            latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
            
            # Restore from backup
            if (latest_backup / "src").exists():
                shutil.copytree(latest_backup / "src", "src", dirs_exist_ok=True)
            
            logger.info(f"Rolled back to version from {latest_backup}")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
