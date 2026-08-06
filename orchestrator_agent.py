from abc import ABC, abstractmethod
from typing import Tuple, List
from openai import OpenAI
from openai.types.responses import Response
from openai.types.responses.response_output_text import AnnotationURLCitation
from langchain_core.documents import Document
from web_search_agent.web_search_agent import WebSearchAgent
from rag_agent.rag_agent import RAGAgent
from ignore_agent.ignore_agent import IgnoreAgent


class Agent(ABC):
    """
    Abstract base class from which any agent inherits.
    """
    def __init__(self):
        self.client = OpenAI()
        self.model = 'gpt-4o-mini'
        self.tools = []

    def generate_response(self, query: str) -> Response:
        """
        Generate a response with this Agent's model.
        """
        response: Response = self.client.responses.create(
            model=self.model,
            tools=self.tools,
            input=query,
        )
        return response


class SubAgent(Agent):
    """
    Abstract base class from which any subagents called on by `OrchestratorAgent` inherit.
    """

    type SubAgentResponse = Tuple[str, List[Document | AnnotationURLCitation]]

    @abstractmethod
    def answer(self, query: str) -> SubAgentResponse:
        pass


class OrchestratorAgent(Agent):
    """
    Orchestrates multiple agents to answer the user query.
    """

    def __init__(self):
        super().__init__()

    def generate_response_string(self, query: str) -> str:
        response = super().generate_response(query)
        return response.output_text

    def _route(self, query: str, max_retries: int = 3) -> type[SubAgent]:
        # Given the user query, returns a SubAgent class
        # that the model chooses to delegate the query to.
        #
        # Returns the class of subagent to use rather than
        # an actual instance because it should ultimately 
        # be up to the client what to do with this 
        # Orchestrator's recommendation.
        current_attempt = 0
        while current_attempt < max_retries:
            routing_prompt = f"""Given the following subagents,
            select the one that is most apt to answer the user query.
            
            Subagents:
            - RAGAgent: Finds any official City of Boston public notices relevant to the user's query
            - WebSearchAgent: Searches the Internet for information relevant to the user's city government-related query
            - IgnoreAgent: Select this agent if the user's query is not related to city government

            User query:
            {query}

            Return the name of the subagent you choose. Return the name and no other text.
            """
            routing_decision = self.generate_response_string(routing_prompt).strip()
            match routing_decision:
                case "RAGAgent":
                    return RAGAgent
                case "WebSearchAgent":
                    return WebSearchAgent
                case "IgnoreAgent":
                    return IgnoreAgent
                case _:
                    continue
            current_attempt += 1
        raise Exception(f"Orchestrator did not produce a valid routing decision in {max_retries} attempts.")

    def answer(self, query: str) -> SubAgent.SubAgentResponse:
        subagent: SubAgent = self._route(query)()
        response = subagent.answer(query)
        return response