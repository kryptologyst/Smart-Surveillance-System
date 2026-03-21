"""Streamlit demo application for Smart Surveillance System."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from omegaconf import DictConfig

from ..utils.core import load_config, setup_logging
from ..surveillance import SurveillanceSystem, DetectionResult
from ..evaluation.benchmarking import ModelEvaluator, EdgePerformanceProfiler


# Configure logging
logger = setup_logging(level="INFO")


class SurveillanceDemo:
    """Streamlit demo for surveillance system."""
    
    def __init__(self, config_path: str):
        """Initialize demo.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.surveillance_system = None
        self.evaluator = None
        self.profiler = None
        
        # Demo state
        self.is_running = False
        self.frame_count = 0
        self.detection_history = []
        self.performance_history = []
        
    def setup_page(self) -> None:
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Smart Surveillance System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add disclaimer
        st.warning(
            "⚠️ **DISCLAIMER**: This is a research/educational demonstration. "
            "NOT FOR SAFETY-CRITICAL DEPLOYMENT. Use at your own risk."
        )
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar controls.
        
        Returns:
            Control parameters
        """
        st.sidebar.title("🔧 Control Panel")
        
        # Model selection
        st.sidebar.subheader("Model Configuration")
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["baseline", "edge_optimized"],
            help="Choose between baseline (higher accuracy) or edge-optimized (lower latency)"
        )
        
        # Detection parameters
        st.sidebar.subheader("Detection Parameters")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        nms_threshold = st.sidebar.slider(
            "NMS Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Non-Maximum Suppression threshold"
        )
        
        # Alert settings
        st.sidebar.subheader("Alert Settings")
        intrusion_enabled = st.sidebar.checkbox(
            "Intrusion Detection",
            value=True,
            help="Enable intrusion zone detection"
        )
        
        loitering_enabled = st.sidebar.checkbox(
            "Loitering Detection",
            value=True,
            help="Enable loitering detection"
        )
        
        loitering_duration = st.sidebar.slider(
            "Loitering Duration (seconds)",
            min_value=10,
            max_value=300,
            value=30,
            step=10,
            help="Minimum duration for loitering alert"
        )
        
        # Performance monitoring
        st.sidebar.subheader("Performance Monitoring")
        show_performance = st.sidebar.checkbox(
            "Show Performance Metrics",
            value=True,
            help="Display real-time performance metrics"
        )
        
        # Data source
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.selectbox(
            "Input Source",
            ["Webcam", "Synthetic", "Upload Video"],
            help="Choose input data source"
        )
        
        return {
            'model_type': model_type,
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'intrusion_enabled': intrusion_enabled,
            'loitering_enabled': loitering_enabled,
            'loitering_duration': loitering_duration,
            'show_performance': show_performance,
            'data_source': data_source
        }
    
    def render_main_content(self, controls: Dict[str, Any]) -> None:
        """Render main content area.
        
        Args:
            controls: Control parameters
        """
        # Title and description
        st.title("🔍 Smart Surveillance System")
        st.markdown(
            "Real-time object detection and surveillance analytics using edge-optimized AI models. "
            "This demo showcases person detection, intrusion monitoring, and loitering detection."
        )
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Detection", "📊 Analytics", "⚡ Performance", "🔧 Model Comparison"])
        
        with tab1:
            self.render_detection_tab(controls)
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_performance_tab()
        
        with tab4:
            self.render_model_comparison_tab()
    
    def render_detection_tab(self, controls: Dict[str, Any]) -> None:
        """Render live detection tab.
        
        Args:
            controls: Control parameters
        """
        st.subheader("Live Object Detection")
        
        # Create columns for video and controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video display
            video_placeholder = st.empty()
            
            # Control buttons
            col_start, col_stop, col_reset = st.columns(3)
            
            with col_start:
                if st.button("▶️ Start Detection", key="start_detection"):
                    self.start_detection(controls)
            
            with col_stop:
                if st.button("⏹️ Stop Detection", key="stop_detection"):
                    self.stop_detection()
            
            with col_reset:
                if st.button("🔄 Reset", key="reset_detection"):
                    self.reset_detection()
        
        with col2:
            # Detection statistics
            st.subheader("Detection Statistics")
            
            if self.detection_history:
                recent_detections = self.detection_history[-10:]  # Last 10 detections
                
                # Count detections by class
                class_counts = {}
                for detection in recent_detections:
                    for class_name in detection.get('class_names', []):
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                if class_counts:
                    st.write("**Recent Detections:**")
                    for class_name, count in class_counts.items():
                        st.write(f"- {class_name}: {count}")
                else:
                    st.write("No recent detections")
            else:
                st.write("No detections yet")
            
            # Alert summary
            st.subheader("Alert Summary")
            alerts = [d for d in self.detection_history if 'alerts' in d and d['alerts']]
            
            if alerts:
                alert_types = {}
                for detection in alerts:
                    for alert in detection['alerts']:
                        alert_type = alert.get('type', 'unknown')
                        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                
                st.write("**Active Alerts:**")
                for alert_type, count in alert_types.items():
                    st.write(f"- {alert_type}: {count}")
            else:
                st.write("No active alerts")
        
        # Simulate detection loop
        if self.is_running:
            self.simulate_detection_loop(video_placeholder, controls)
    
    def render_analytics_tab(self) -> None:
        """Render analytics tab."""
        st.subheader("Detection Analytics")
        
        if not self.detection_history:
            st.info("Start detection to see analytics")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.detection_history)
        
        # Time series of detections
        st.subheader("Detection Timeline")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Plot detection count over time
            detection_counts = df.groupby(df['timestamp'].dt.floor('T')).size()
            
            fig = px.line(
                x=detection_counts.index,
                y=detection_counts.values,
                title="Detections Over Time",
                labels={'x': 'Time', 'y': 'Number of Detections'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detection distribution
        st.subheader("Detection Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            if 'class_names' in df.columns:
                all_classes = []
                for classes in df['class_names']:
                    if isinstance(classes, list):
                        all_classes.extend(classes)
                
                if all_classes:
                    class_counts = pd.Series(all_classes).value_counts()
                    
                    fig = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="Detection Class Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            if 'scores' in df.columns:
                all_scores = []
                for scores in df['scores']:
                    if isinstance(scores, list):
                        all_scores.extend(scores)
                
                if all_scores:
                    fig = px.histogram(
                        x=all_scores,
                        title="Confidence Score Distribution",
                        labels={'x': 'Confidence Score', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_tab(self) -> None:
        """Render performance monitoring tab."""
        st.subheader("Performance Metrics")
        
        if not self.performance_history:
            st.info("Start detection to see performance metrics")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_history)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_fps = df['fps'].mean() if 'fps' in df.columns else 0
            st.metric("Average FPS", f"{avg_fps:.1f}")
        
        with col2:
            avg_latency = df['latency_ms'].mean() if 'latency_ms' in df.columns else 0
            st.metric("Average Latency", f"{avg_latency:.1f} ms")
        
        with col3:
            avg_memory = df['memory_mb'].mean() if 'memory_mb' in df.columns else 0
            st.metric("Average Memory", f"{avg_memory:.1f} MB")
        
        with col4:
            total_frames = len(df)
            st.metric("Total Frames", f"{total_frames}")
        
        # Performance charts
        st.subheader("Performance Over Time")
        
        if len(df) > 1:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('FPS', 'Latency (ms)', 'Memory (MB)', 'CPU Usage (%)'),
                vertical_spacing=0.1
            )
            
            # FPS
            if 'fps' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['fps'], name='FPS'),
                    row=1, col=1
                )
            
            # Latency
            if 'latency_ms' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['latency_ms'], name='Latency'),
                    row=1, col=2
                )
            
            # Memory
            if 'memory_mb' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['memory_mb'], name='Memory'),
                    row=2, col=1
                )
            
            # CPU (simulated)
            if 'cpu_usage' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['cpu_usage'], name='CPU'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_comparison_tab(self) -> None:
        """Render model comparison tab."""
        st.subheader("Model Performance Comparison")
        
        # Simulated comparison data
        comparison_data = {
            'Model': ['YOLOv8n (Baseline)', 'YOLOv8n Quantized', 'Edge Optimized', 'Custom Tiny'],
            'mAP': [0.45, 0.42, 0.38, 0.35],
            'FPS': [25, 35, 45, 60],
            'Latency (ms)': [40, 28, 22, 17],
            'Memory (MB)': [800, 600, 400, 300],
            'Model Size (MB)': [6.2, 3.1, 2.5, 1.8]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(df, use_container_width=True)
        
        # Performance vs Accuracy plot
        st.subheader("Performance vs Accuracy Trade-off")
        
        fig = px.scatter(
            df,
            x='Latency (ms)',
            y='mAP',
            size='Model Size (MB)',
            color='FPS',
            hover_name='Model',
            title='Model Performance Comparison',
            labels={'Latency (ms)': 'Latency (ms)', 'mAP': 'mAP Score'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("Detailed Performance Metrics")
        
        metrics = ['mAP', 'FPS', 'Latency (ms)', 'Memory (MB)', 'Model Size (MB)']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Model'],
                y=df[metric],
                text=df[metric],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Performance Metrics Comparison',
            xaxis_title='Model',
            yaxis_title='Value',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def start_detection(self, controls: Dict[str, Any]) -> None:
        """Start detection process.
        
        Args:
            controls: Control parameters
        """
        try:
            # Initialize surveillance system
            self.surveillance_system = SurveillanceSystem(self.config_path)
            
            # Update configuration based on controls
            self.surveillance_system.config.model.baseline.confidence_threshold = controls['confidence_threshold']
            self.surveillance_system.config.model.baseline.nms_threshold = controls['nms_threshold']
            self.surveillance_system.config.detection.alerts.intrusion_zones.enabled = controls['intrusion_enabled']
            self.surveillance_system.config.detection.alerts.loitering.enabled = controls['loitering_enabled']
            self.surveillance_system.config.detection.alerts.loitering.min_duration_seconds = controls['loitering_duration']
            
            self.is_running = True
            st.success("Detection started successfully!")
            
        except Exception as e:
            st.error(f"Error starting detection: {e}")
            logger.error(f"Error starting detection: {e}")
    
    def stop_detection(self) -> None:
        """Stop detection process."""
        self.is_running = False
        if self.surveillance_system:
            asyncio.run(self.surveillance_system.stop())
        st.info("Detection stopped")
    
    def reset_detection(self) -> None:
        """Reset detection state."""
        self.stop_detection()
        self.detection_history = []
        self.performance_history = []
        self.frame_count = 0
        st.info("Detection state reset")
    
    def simulate_detection_loop(self, video_placeholder, controls: Dict[str, Any]) -> None:
        """Simulate detection loop for demo.
        
        Args:
            video_placeholder: Streamlit placeholder for video
            controls: Control parameters
        """
        # Generate synthetic frame
        frame = self.generate_synthetic_frame()
        
        # Simulate detection
        detections = self.simulate_detections(frame, controls)
        
        # Draw detections on frame
        result_frame = self.draw_detections_on_frame(frame, detections)
        
        # Display frame
        video_placeholder.image(result_frame, channels="BGR", use_column_width=True)
        
        # Update history
        self.update_detection_history(detections)
        self.update_performance_history()
        
        # Small delay
        time.sleep(0.1)
    
    def generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic frame for demo.
        
        Returns:
            Synthetic frame
        """
        # Create base frame
        frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        
        # Add moving objects
        if self.frame_count % 60 < 30:  # Person appears for 30 frames
            center_x = int(320 + (self.frame_count % 60) * 5)
            center_y = int(240)
            
            # Draw person
            cv2.rectangle(frame, 
                         (center_x - 20, center_y - 40), 
                         (center_x + 20, center_y + 40), 
                         (100, 100, 100), -1)
            cv2.circle(frame, (center_x, center_y - 50), 15, (200, 180, 160), -1)
        
        self.frame_count += 1
        return frame
    
    def simulate_detections(self, frame: np.ndarray, controls: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate detection results.
        
        Args:
            frame: Input frame
            controls: Control parameters
            
        Returns:
            Simulated detection results
        """
        # Simulate detections based on frame content
        detections = {
            'boxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': [],
            'timestamp': time.time()
        }
        
        # Add some random detections
        if np.random.random() > 0.3:  # 70% chance of detection
            detections['boxes'] = [[100, 100, 200, 300]]
            detections['scores'] = [0.85]
            detections['class_ids'] = [0]  # person
            detections['class_names'] = ['person']
        
        # Simulate alerts
        alerts = []
        if controls['intrusion_enabled'] and np.random.random() > 0.8:
            alerts.append({
                'type': 'intrusion',
                'zone_id': 'zone_1',
                'confidence': 0.9,
                'timestamp': time.time()
            })
        
        if controls['loitering_enabled'] and np.random.random() > 0.9:
            alerts.append({
                'type': 'loitering',
                'duration': 45,
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        detections['alerts'] = alerts
        
        return detections
    
    def draw_detections_on_frame(self, frame: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
        """Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: Detection results
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Draw bounding boxes
        for i, (box, score, class_name) in enumerate(zip(
            detections['boxes'], detections['scores'], detections['class_names']
        )):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw alert indicators
        if detections.get('alerts'):
            for alert in detections['alerts']:
                alert_text = f"ALERT: {alert['type'].upper()}"
                cv2.putText(result_frame, alert_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_frame
    
    def update_detection_history(self, detections: Dict[str, Any]) -> None:
        """Update detection history.
        
        Args:
            detections: Detection results
        """
        self.detection_history.append(detections)
        
        # Keep only last 100 detections
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
    
    def update_performance_history(self) -> None:
        """Update performance history."""
        # Simulate performance metrics
        performance = {
            'fps': np.random.normal(25, 5),
            'latency_ms': np.random.normal(40, 10),
            'memory_mb': np.random.normal(500, 50),
            'cpu_usage': np.random.normal(60, 10),
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance)
        
        # Keep only last 100 measurements
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def run(self) -> None:
        """Run the demo application."""
        self.setup_page()
        
        # Render sidebar
        controls = self.render_sidebar()
        
        # Render main content
        self.render_main_content(controls)


def main():
    """Main function to run the demo."""
    config_path = "configs/device/surveillance.yaml"
    
    demo = SurveillanceDemo(config_path)
    demo.run()


if __name__ == "__main__":
    main()
