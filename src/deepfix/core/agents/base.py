from abc import ABC, abstractmethod
from typing import List, Type
import dspy

from ..artifacts import Artifacts
from .models import AgentContext, AgentResult
from ..query.intelligence import IntelligenceClient

class Agent(dspy.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, context: AgentContext) -> AgentResult:
        pass

class ArtifactAnalyzer(Agent):

    def __init__(self,):
        self.agent_name = self.__class__.__name__.lower().replace('agent', '')
    
    @abstractmethod
    def _run(self, context: AgentContext) -> AgentResult:
        pass    
    
    def forward(self, context: AgentContext) -> AgentResult:
        return self._run(context)

    @property
    def supported_artifact_types(self) -> List[Type[Artifacts]]:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
