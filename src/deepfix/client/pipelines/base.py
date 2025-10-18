from abc import ABC, abstractmethod

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class Step(ABC):
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def run(self, *args, context: dict, **kwargs) -> dict:
        pass
    
    @property
    def name(self) -> str:
        return self.get_name()


class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.context = {}

    def run(self, **kwargs) -> dict:
        """Execute the pipeline with provided context.
        
        Args:
            **kwargs: Initial context values to pass to steps
            
        Returns:
            dict: The accumulated context dictionary after all steps complete
            
        Raises:
            Exception: Re-raises any exception from a step after logging with traceback
        """
        self.context.update(kwargs)
        for step in self.steps:
            try:
                step.run(context=self.context)
            except Exception as e:
                LOGGER.error("Step %s failed with error: %s", step.__class__.__name__, e, exc_info=True)
                raise
        return self.context
