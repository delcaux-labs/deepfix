from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..models import IntelligenceResponse, Capabilities


class BaseProvider(ABC):
    @abstractmethod
    def execute(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> IntelligenceResponse:
        pass

    @abstractmethod
    def get_capabilities(self) -> List[Capabilities]:
        pass
