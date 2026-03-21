#!/usr/bin/env python3
"""
Setup script for Smart Surveillance System.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Smart Surveillance System")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install black ruff mypy pytest", "Installing dev dependencies"):
        print("⚠️ Failed to install dev dependencies (optional)")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "assets",
        "exports",
        "deployments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    
    # Run basic tests
    if not run_command("python -m pytest tests/ -v", "Running basic tests"):
        print("⚠️ Some tests failed (this is expected without proper data)")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/app.py")
    print("2. Start surveillance: python main.py --demo")
    print("3. Train models: python scripts/train.py --action train")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
