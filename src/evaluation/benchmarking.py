"""Evaluation and benchmarking system for surveillance models."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

from ..utils.core import PerformanceTimer, get_device, logger
from ..models.detection import EdgeOptimizedYOLO, QuantizedModel, benchmark_model


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics container."""
        self.metrics = defaultdict(list)
        self.confusion_matrix = None
        self.class_metrics = {}
    
    def add_metric(self, name: str, value: float) -> None:
        """Add metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """Get average value for metric.
        
        Args:
            name: Metric name
            
        Returns:
            Average metric value
        """
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_std(self, name: str) -> float:
        """Get standard deviation for metric.
        
        Args:
            name: Metric name
            
        Returns:
            Standard deviation
        """
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return np.std(self.metrics[name])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {}
        for name, values in self.metrics.items():
            result[f"{name}_mean"] = np.mean(values)
            result[f"{name}_std"] = np.std(values)
            result[f"{name}_min"] = np.min(values)
            result[f"{name}_max"] = np.max(values)
        return result


class ModelEvaluator:
    """Model evaluation and benchmarking system."""
    
    def __init__(self, config: DictConfig):
        """Initialize model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = get_device()
        self.results = {}
        
    def evaluate_model(self, model: Any, test_data: List[Dict[str, Any]], 
                      model_name: str) -> Dict[str, Any]:
        """Evaluate model on test data.
        
        Args:
            model: Model to evaluate
            test_data: List of test samples with 'image' and 'annotations'
            model_name: Name of the model
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = EvaluationMetrics()
        all_predictions = []
        all_ground_truth = []
        
        with PerformanceTimer(f"Model evaluation: {model_name}", logger):
            for i, sample in enumerate(test_data):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}/{len(test_data)}")
                
                # Run inference
                start_time = time.perf_counter()
                predictions = self._run_inference(model, sample['image'])
                inference_time = time.perf_counter() - start_time
                
                # Calculate metrics
                sample_metrics = self._calculate_sample_metrics(
                    predictions, sample['annotations']
                )
                
                # Add to overall metrics
                for metric_name, value in sample_metrics.items():
                    metrics.add_metric(metric_name, value)
                
                metrics.add_metric('inference_time_ms', inference_time * 1000)
                
                all_predictions.extend(predictions)
                all_ground_truth.extend(sample['annotations'])
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            all_predictions, all_ground_truth
        )
        
        # Combine metrics
        result = {
            'model_name': model_name,
            'sample_metrics': metrics.to_dict(),
            'overall_metrics': overall_metrics,
            'num_samples': len(test_data)
        }
        
        self.results[model_name] = result
        return result
    
    def _run_inference(self, model: Any, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on single image.
        
        Args:
            model: Model to run inference with
            image: Input image
            
        Returns:
            List of predictions
        """
        # This is a simplified implementation
        # In practice, this would depend on the specific model interface
        
        if hasattr(model, 'predict'):
            # For models with predict method
            results = model.predict(image)
            return self._convert_predictions(results)
        else:
            # For PyTorch models
            import torch
            
            # Preprocess image
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.float() / 255.0
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            return self._convert_predictions(output)
    
    def _convert_predictions(self, raw_predictions: Any) -> List[Dict[str, Any]]:
        """Convert raw model predictions to standard format.
        
        Args:
            raw_predictions: Raw model output
            
        Returns:
            List of predictions in standard format
        """
        # Simplified conversion - would need actual implementation
        # based on model output format
        return []
    
    def _calculate_sample_metrics(self, predictions: List[Dict[str, Any]], 
                                 ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for single sample.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
            
        Returns:
            Sample metrics
        """
        # Calculate precision, recall, F1 for this sample
        if not predictions and not ground_truth:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not predictions:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate IoU between predictions and ground truth
        ious = []
        for pred in predictions:
            max_iou = 0.0
            for gt in ground_truth:
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
        
        # Calculate metrics based on IoU threshold
        iou_threshold = 0.5
        true_positives = sum(1 for iou in ious if iou >= iou_threshold)
        
        precision = true_positives / len(predictions) if predictions else 0.0
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': np.mean(ious) if ious else 0.0
        }
    
    def _calculate_overall_metrics(self, predictions: List[Dict[str, Any]], 
                                  ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall metrics across all samples.
        
        Args:
            predictions: All predictions
            ground_truth: All ground truth annotations
            
        Returns:
            Overall metrics
        """
        # Calculate mAP (mean Average Precision)
        map_score = self._calculate_map(predictions, ground_truth)
        
        # Calculate overall precision, recall, F1
        if not predictions and not ground_truth:
            precision = recall = f1 = 1.0
        elif not predictions:
            precision = recall = f1 = 0.0
        elif not ground_truth:
            precision = recall = f1 = 0.0
        else:
            # Calculate overall metrics
            true_positives = 0
            false_positives = 0
            false_negatives = len(ground_truth)
            
            for pred in predictions:
                matched = False
                for gt in ground_truth:
                    if self._calculate_iou(pred['bbox'], gt['bbox']) >= 0.5:
                        true_positives += 1
                        false_negatives -= 1
                        matched = True
                        break
                
                if not matched:
                    false_positives += 1
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'map': map_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth)
        }
    
    def _calculate_map(self, predictions: List[Dict[str, Any]], 
                      ground_truth: List[Dict[str, Any]]) -> float:
        """Calculate mean Average Precision (mAP).
        
        Args:
            predictions: All predictions
            ground_truth: All ground truth annotations
            
        Returns:
            mAP score
        """
        # Simplified mAP calculation
        # In practice, this would be more sophisticated
        
        if not predictions or not ground_truth:
            return 0.0
        
        # Group by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for pred in predictions:
            pred_by_class[pred['class']].append(pred)
        
        for gt in ground_truth:
            gt_by_class[gt['class']].append(gt)
        
        # Calculate AP for each class
        aps = []
        for class_name in pred_by_class.keys():
            if class_name not in gt_by_class:
                continue
            
            class_preds = pred_by_class[class_name]
            class_gts = gt_by_class[class_name]
            
            # Sort predictions by confidence
            class_preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision-recall curve
            precisions = []
            recalls = []
            
            for i in range(len(class_preds)):
                tp = 0
                fp = 0
                
                for j in range(i + 1):
                    pred = class_preds[j]
                    matched = False
                    
                    for gt in class_gts:
                        if self._calculate_iou(pred['bbox'], gt['bbox']) >= 0.5:
                            tp += 1
                            matched = True
                            break
                    
                    if not matched:
                        fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / len(class_gts) if class_gts else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate AP using 11-point interpolation
            ap = self._calculate_ap_11_point(precisions, recalls)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def _calculate_ap_11_point(self, precisions: List[float], 
                              recalls: List[float]) -> float:
        """Calculate AP using 11-point interpolation.
        
        Args:
            precisions: Precision values
            recalls: Recall values
            
        Returns:
            Average Precision
        """
        if not precisions or not recalls:
            return 0.0
        
        # 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        max_precisions = []
        
        for threshold in recall_thresholds:
            max_precision = 0.0
            for i, recall in enumerate(recalls):
                if recall >= threshold:
                    max_precision = max(max_precision, precisions[i])
            max_precisions.append(max_precision)
        
        return np.mean(max_precisions)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def benchmark_models(self, models: Dict[str, Any], 
                        test_data: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Benchmark multiple models for performance.
        
        Args:
            models: Dictionary of model name to model instance
            test_data: List of test images
            
        Returns:
            Benchmark results
        """
        logger.info("Starting model benchmarking")
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Benchmarking model: {model_name}")
            
            # Run benchmark
            metrics = benchmark_model(model, test_data)
            benchmark_results[model_name] = metrics
            
            logger.info(f"Benchmark results for {model_name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return benchmark_results
    
    def create_leaderboard(self, output_path: Union[str, Path]) -> None:
        """Create performance leaderboard.
        
        Args:
            output_path: Output file path
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        # Create leaderboard data
        leaderboard_data = []
        
        for model_name, result in self.results.items():
            row = {
                'Model': model_name,
                'mAP': result['overall_metrics']['map'],
                'Precision': result['overall_metrics']['precision'],
                'Recall': result['overall_metrics']['recall'],
                'F1': result['overall_metrics']['f1'],
                'Avg Latency (ms)': result['sample_metrics']['inference_time_ms_mean'],
                'Std Latency (ms)': result['sample_metrics']['inference_time_ms_std'],
                'FPS': 1000 / result['sample_metrics']['inference_time_ms_mean'] if result['sample_metrics']['inference_time_ms_mean'] > 0 else 0,
                'Samples': result['num_samples']
            }
            leaderboard_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by mAP (descending)
        df = df.sort_values('mAP', ascending=False)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Create visualization
        self._create_leaderboard_plots(df, Path(output_path).parent)
        
        logger.info(f"Leaderboard saved to: {output_path}")
    
    def _create_leaderboard_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create leaderboard visualization plots.
        
        Args:
            df: Leaderboard DataFrame
            output_dir: Output directory
        """
        plt.style.use('seaborn-v0_8')
        
        # Accuracy vs Latency plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(df['Avg Latency (ms)'], df['mAP'], 
                           s=100, alpha=0.7, c=df['FPS'], 
                           cmap='viridis')
        
        ax.set_xlabel('Average Latency (ms)')
        ax.set_ylabel('mAP')
        ax.set_title('Model Performance: Accuracy vs Latency')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('FPS')
        
        # Add model labels
        for i, model in enumerate(df['Model']):
            ax.annotate(model, (df['Avg Latency (ms)'].iloc[i], df['mAP'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_vs_latency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance comparison bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy metrics
        metrics = ['mAP', 'Precision', 'Recall', 'F1']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, df[metric], width, label=metric)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Accuracy Metrics Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(df['Model'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance metrics
        perf_metrics = ['FPS', 'Avg Latency (ms)']
        x = np.arange(len(df))
        width = 0.35
        
        ax2.bar(x - width/2, df['FPS'], width, label='FPS', alpha=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + width/2, df['Avg Latency (ms)'], width, 
                    label='Latency (ms)', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('FPS')
        ax2_twin.set_ylabel('Latency (ms)')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Model'], rotation=45)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Leaderboard plots saved to: {output_dir}")


class EdgePerformanceProfiler:
    """Profile edge device performance."""
    
    def __init__(self, config: DictConfig):
        """Initialize edge performance profiler.
        
        Args:
            config: Profiling configuration
        """
        self.config = config
        self.metrics = defaultdict(list)
    
    def profile_inference(self, model: Any, test_data: List[np.ndarray]) -> Dict[str, Any]:
        """Profile model inference performance.
        
        Args:
            model: Model to profile
            test_data: Test data
            
        Returns:
            Performance metrics
        """
        logger.info("Starting inference profiling")
        
        # Warmup
        for i in range(10):
            if i < len(test_data):
                _ = self._run_inference(model, test_data[i])
        
        # Profile
        times = []
        memory_usage = []
        
        for i, data in enumerate(test_data):
            if i % 50 == 0:
                logger.info(f"Profiling sample {i}/{len(test_data)}")
            
            start_time = time.perf_counter()
            _ = self._run_inference(model, data)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Get memory usage (simplified)
            import psutil
            memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
        
        # Calculate statistics
        times = np.array(times)
        memory_usage = np.array(memory_usage)
        
        metrics = {
            'latency_mean_ms': np.mean(times) * 1000,
            'latency_std_ms': np.std(times) * 1000,
            'latency_p50_ms': np.percentile(times, 50) * 1000,
            'latency_p95_ms': np.percentile(times, 95) * 1000,
            'latency_p99_ms': np.percentile(times, 99) * 1000,
            'fps_mean': 1.0 / np.mean(times),
            'memory_mean_mb': np.mean(memory_usage),
            'memory_max_mb': np.max(memory_usage),
            'throughput_samples_per_sec': len(test_data) / np.sum(times)
        }
        
        return metrics
    
    def _run_inference(self, model: Any, data: np.ndarray) -> Any:
        """Run single inference."""
        if hasattr(model, 'predict'):
            return model.predict(data)
        else:
            import torch
            input_tensor = torch.from_numpy(data).unsqueeze(0)
            with torch.no_grad():
                return model(input_tensor)
    
    def profile_energy_consumption(self, model: Any, test_data: List[np.ndarray]) -> Dict[str, float]:
        """Profile energy consumption (simplified simulation).
        
        Args:
            model: Model to profile
            test_data: Test data
            
        Returns:
            Energy consumption metrics
        """
        logger.info("Profiling energy consumption")
        
        # This is a simplified energy profiling
        # In practice, you would use hardware-specific tools
        
        total_time = 0.0
        for data in test_data:
            start_time = time.perf_counter()
            _ = self._run_inference(model, data)
            total_time += time.perf_counter() - start_time
        
        # Estimate energy consumption (simplified)
        # These values would be device-specific
        cpu_power_w = 5.0  # Watts
        gpu_power_w = 15.0  # Watts
        
        total_energy_j = (cpu_power_w + gpu_power_w) * total_time
        energy_per_inference_j = total_energy_j / len(test_data)
        
        return {
            'total_energy_j': total_energy_j,
            'energy_per_inference_j': energy_per_inference_j,
            'energy_per_inference_mwh': energy_per_inference_j * 1000 / 3.6,  # Convert to mWh
            'total_time_s': total_time,
            'avg_power_w': (cpu_power_w + gpu_power_w)
        }
