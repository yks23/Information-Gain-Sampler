"""
Base benchmark interface for all evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks.
    
    Subclasses must implement:
        - load_data(): Load and return the dataset
        - build_prompt(): Build a prompt from a data sample
        - evaluate(): Evaluate predictions against references
    """

    def __init__(self, task_name: str):
        self.task_name = task_name

    @abstractmethod
    def load_data(self, data_path: str) -> List[Dict]:
        """
        Load the dataset from a file path.
        
        Args:
            data_path: Path to the dataset file (JSON, JSONL, etc.)
            
        Returns:
            List of data samples (dicts)
        """
        ...

    @abstractmethod
    def build_prompt(self, sample: Dict, use_shot: bool = True) -> str:
        """
        Build a prompt from a data sample.
        
        Args:
            sample: A single data sample (dict)
            use_shot: Whether to include few-shot examples
            
        Returns:
            The formatted prompt string
        """
        ...

    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[Dict]) -> Dict[str, float]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: List of model-generated outputs
            references: List of reference data samples
            
        Returns:
            Dictionary of metric names to scores
        """
        ...

    def run(
        self,
        model_adapter,
        generator,
        data_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.
        
        This is a convenience method that loads data, generates predictions,
        and evaluates them. Subclasses can override for custom behavior.
        
        Args:
            model_adapter: Model adapter instance
            generator: Generation function
            data_path: Path to dataset
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing results and metrics
        """
        dataset = self.load_data(data_path)
        predictions = []
        
        for sample in dataset:
            prompt = self.build_prompt(sample, use_shot=kwargs.get('use_shot', True))
            # Generate prediction (implementation depends on generator)
            # This is a placeholder - actual implementation will vary
            prediction = generator(model_adapter, prompt, **kwargs)
            predictions.append(prediction)
        
        metrics = self.evaluate(predictions, dataset)
        return {
            'predictions': predictions,
            'metrics': metrics,
        }

