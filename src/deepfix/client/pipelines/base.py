from abc import ABC, abstractmethod

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class Step(ABC):
    
    def get_name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def run(self, *args, context: dict, **kwargs) -> dict:
        pass


class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.context = {}

    def run(self, **kwargs) -> dict:
        self.context.update(kwargs)
        for step in self.steps:
            try:
                step.run(context=self.context)
            except Exception as e:
                LOGGER.error(f"Step {step.__class__.__name__} failed with error: {e}")
        return self.context
