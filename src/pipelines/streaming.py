"""Data pipeline and streaming components for surveillance system."""

import asyncio
import logging
import queue
import threading
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import websockets
from omegaconf import DictConfig

from ..utils.core import PerformanceTimer, validate_image, logger


class CameraStream:
    """Camera stream handler with edge constraints."""
    
    def __init__(self, source: Union[int, str], config: DictConfig):
        """Initialize camera stream.
        
        Args:
            source: Camera source (0 for default, path for video file)
            config: Stream configuration
        """
        self.source = source
        self.config = config
        self.cap = None
        self.frame_buffer = deque(maxlen=config.streaming.camera.buffer_size)
        self.is_running = False
        self.frame_count = 0
        
    def start(self) -> bool:
        """Start camera stream.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.streaming.camera.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.streaming.camera.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.streaming.camera.fps)
            
            self.is_running = True
            logger.info(f"Camera stream started: {self.source}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera stream: {e}")
            return False
    
    def stop(self) -> None:
        """Stop camera stream."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera stream stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from camera.
        
        Returns:
            Frame array or None if no frame available
        """
        if not self.is_running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Validate frame
        if not validate_image(frame):
            logger.warning(f"Invalid frame received: {self.frame_count}")
            return None
        
        # Add to buffer
        self.frame_buffer.append(frame)
        
        return frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from buffer.
        
        Returns:
            Latest frame or None if buffer empty
        """
        if self.frame_buffer:
            return self.frame_buffer[-1]
        return None


class MQTTClient:
    """MQTT client for surveillance data streaming."""
    
    def __init__(self, config: DictConfig):
        """Initialize MQTT client.
        
        Args:
            config: MQTT configuration
        """
        self.config = config
        self.client = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        
    def connect(self) -> bool:
        """Connect to MQTT broker.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client = mqtt.Client()
            
            # Set credentials if provided
            if self.config.mqtt.username and self.config.mqtt.password:
                self.client.username_pw_set(
                    self.config.mqtt.username, 
                    self.config.mqtt.password
                )
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            # Connect to broker
            self.client.connect(
                self.config.mqtt.broker_host,
                self.config.mqtt.broker_port,
                self.config.mqtt.keepalive
            )
            
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                logger.info("MQTT client connected successfully")
                return True
            else:
                logger.error("MQTT connection timeout")
                return False
                
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.is_connected = False
        logger.info("MQTT client disconnected")
    
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """MQTT connection callback."""
        if rc == 0:
            self.is_connected = True
            logger.info("MQTT broker connected")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
    
    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """MQTT disconnection callback."""
        self.is_connected = False
        logger.info("MQTT broker disconnected")
    
    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """MQTT message callback."""
        try:
            message = {
                'topic': msg.topic,
                'payload': msg.payload.decode('utf-8'),
                'qos': msg.qos,
                'retain': msg.retain,
                'timestamp': time.time()
            }
            self.message_queue.put(message)
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def publish_detection(self, detection_data: Dict[str, Any]) -> None:
        """Publish detection data to MQTT.
        
        Args:
            detection_data: Detection data to publish
        """
        if not self.is_connected:
            logger.warning("MQTT client not connected")
            return
        
        try:
            import json
            payload = json.dumps(detection_data)
            
            self.client.publish(
                self.config.mqtt.topics.detections,
                payload,
                qos=self.config.mqtt.qos,
                retain=self.config.mqtt.retain
            )
            
        except Exception as e:
            logger.error(f"Error publishing detection: {e}")
    
    def publish_alert(self, alert_data: Dict[str, Any]) -> None:
        """Publish alert data to MQTT.
        
        Args:
            alert_data: Alert data to publish
        """
        if not self.is_connected:
            logger.warning("MQTT client not connected")
            return
        
        try:
            import json
            payload = json.dumps(alert_data)
            
            self.client.publish(
                self.config.mqtt.topics.alerts,
                payload,
                qos=self.config.mqtt.qos,
                retain=self.config.mqtt.retain
            )
            
        except Exception as e:
            logger.error(f"Error publishing alert: {e}")


class WebSocketServer:
    """WebSocket server for real-time streaming."""
    
    def __init__(self, config: DictConfig):
        """Initialize WebSocket server.
        
        Args:
            config: WebSocket configuration
        """
        self.config = config
        self.clients = set()
        self.server = None
        
    async def start(self) -> None:
        """Start WebSocket server."""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.config.websocket.host,
                self.config.websocket.port,
                max_size=2**20,  # 1MB max message size
                compression=self.config.websocket.compression
            )
            
            logger.info(f"WebSocket server started on {self.config.websocket.host}:{self.config.websocket.port}")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle WebSocket client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        self.clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                # Handle client messages if needed
                await self.process_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
    
    async def process_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str) -> None:
        """Process message from WebSocket client.
        
        Args:
            websocket: WebSocket connection
            message: Client message
        """
        try:
            import json
            data = json.loads(message)
            
            # Handle different message types
            if data.get('type') == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
            
        except Exception as e:
            logger.error(f"Error processing client message: {e}")
    
    async def broadcast_frame(self, frame_data: bytes) -> None:
        """Broadcast frame data to all connected clients.
        
        Args:
            frame_data: Encoded frame data
        """
        if not self.clients:
            return
        
        # Create message
        message = {
            'type': 'frame',
            'data': frame_data.decode('latin-1'),  # Convert bytes to string
            'timestamp': time.time()
        }
        
        import json
        payload = json.dumps(message)
        
        # Send to all clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(payload)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients


class DataPipeline:
    """Main data pipeline for surveillance system."""
    
    def __init__(self, config: DictConfig):
        """Initialize data pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.camera_stream = None
        self.mqtt_client = None
        self.websocket_server = None
        self.is_running = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Camera stream
        if self.config.streaming.camera.source is not None:
            self.camera_stream = CameraStream(
                self.config.streaming.camera.source,
                self.config
            )
        
        # MQTT client
        if self.config.mqtt.enabled:
            self.mqtt_client = MQTTClient(self.config)
        
        # WebSocket server
        if self.config.websocket.enabled:
            self.websocket_server = WebSocketServer(self.config)
    
    async def start(self) -> bool:
        """Start data pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Start camera stream
            if self.camera_stream:
                if not self.camera_stream.start():
                    logger.error("Failed to start camera stream")
                    return False
            
            # Connect MQTT client
            if self.mqtt_client:
                if not self.mqtt_client.connect():
                    logger.error("Failed to connect MQTT client")
                    return False
            
            # Start WebSocket server
            if self.websocket_server:
                await self.websocket_server.start()
            
            self.is_running = True
            logger.info("Data pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting data pipeline: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop data pipeline."""
        self.is_running = False
        
        # Stop camera stream
        if self.camera_stream:
            self.camera_stream.stop()
        
        # Disconnect MQTT client
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        
        # Stop WebSocket server
        if self.websocket_server:
            await self.websocket_server.stop()
        
        logger.info("Data pipeline stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from pipeline.
        
        Returns:
            Frame array or None if no frame available
        """
        if self.camera_stream:
            return self.camera_stream.get_frame()
        return None
    
    def publish_detection(self, detection_data: Dict[str, Any]) -> None:
        """Publish detection data.
        
        Args:
            detection_data: Detection data to publish
        """
        if self.mqtt_client:
            self.mqtt_client.publish_detection(detection_data)
    
    def publish_alert(self, alert_data: Dict[str, Any]) -> None:
        """Publish alert data.
        
        Args:
            alert_data: Alert data to publish
        """
        if self.mqtt_client:
            self.mqtt_client.publish_alert(alert_data)
    
    async def broadcast_frame(self, frame: np.ndarray) -> None:
        """Broadcast frame to WebSocket clients.
        
        Args:
            frame: Frame to broadcast
        """
        if self.websocket_server:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_data = buffer.tobytes()
            
            await self.websocket_server.broadcast_frame(frame_data)


class SyntheticDataGenerator:
    """Generate synthetic data for testing and development."""
    
    def __init__(self, config: DictConfig):
        """Initialize synthetic data generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        self.frame_count = 0
        
    def generate_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generate synthetic video frame.
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            Synthetic frame array
        """
        # Create base frame
        frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        
        # Add moving objects (simulated people)
        if self.frame_count % 30 < 15:  # Person appears for 15 frames
            # Draw a simple person-like shape
            center_x = int(width * 0.3 + (self.frame_count % 30) * 10)
            center_y = int(height * 0.6)
            
            # Body
            cv2.rectangle(frame, 
                         (center_x - 20, center_y - 40), 
                         (center_x + 20, center_y + 40), 
                         (100, 100, 100), -1)
            
            # Head
            cv2.circle(frame, (center_x, center_y - 50), 15, (200, 180, 160), -1)
        
        # Add noise
        noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        self.frame_count += 1
        return frame
    
    def generate_audio_sample(self, duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic audio sample.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio sample array
        """
        samples = int(duration * sample_rate)
        
        # Generate simple sine wave with noise
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 note
        signal = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        noise = np.random.normal(0, 0.1, samples)
        signal += noise
        
        return signal.astype(np.float32)
