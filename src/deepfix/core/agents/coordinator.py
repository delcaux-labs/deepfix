from ..query.knowledgebase.kb import KnowledgeBridge
from ..agents.models import AgentContext
from ..agents.synthesizer import NarrativeSynthesizer
from ..query.intelligence.models import IntelligenceConfig
from ..agents.models import AgentContext
from ..agents.base import Agent


class AgentCoordinator:
    def __init__(self, intelligence_config: IntelligenceConfig):
        self.agents = self._load_agents()
        self.knowledge_bridge = KnowledgeBridge(intelligence_config)
    
    def run(self, context: dict) -> dict:
        agent_context = AgentContext.from_pipeline_context(context)
        
        # Run applicable agents sequentially
        for agent in self._get_applicable_agents(agent_context):
            try:
                result = agent.run(agent_context)
                agent_context.agent_results[agent.name] = result
                agent_context.completed_agents.append(agent.name)
            except Exception as e:
                self._handle_agent_failure(agent, e, agent_context)
        
        # Synthesize results
        synthesizer = NarrativeSynthesizer(self.knowledge_bridge)
        advisor_result = synthesizer.synthesize(agent_context)
        
        context["advisor_result"] = advisor_result
        return context
    
    def _load_agents(self):
        pass
    
    def _get_applicable_agents(self, agent_context: AgentContext):
        pass
    
    def _handle_agent_failure(self, agent: Agent, e: Exception, agent_context: AgentContext):
        pass