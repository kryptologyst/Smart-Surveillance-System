#!/usr/bin/env python3
"""
Main entry point for Smart Surveillance System.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.core import setup_logging, load_config
from src.surveillance import SurveillanceSystem


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Smart Surveillance System")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/device/surveillance.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic data"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize surveillance system
        system = SurveillanceSystem(args.config)
        
        # Start system
        async def run_system():
            if await system.start():
                logger.info("Surveillance system started successfully")
                
                if args.demo:
                    logger.info("Running in demo mode")
                    # Run with synthetic data
                    await system.run_surveillance_loop()
                else:
                    # Run normal surveillance loop
                    await system.run_surveillance_loop()
            else:
                logger.error("Failed to start surveillance system")
                return False
        
        # Run the system
        asyncio.run(run_system())
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"Error running surveillance system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
