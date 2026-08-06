from orchestrator_agent import SubAgent
from openai.types.responses import Response

class WebSearchAgent(SubAgent):

    def __init__(self, model):
        super().__init__()
        self.tools.append({"type": "web_search"})

    def answer(self, query: str) -> SubAgent.SubAgentResponse:
        response: Response = self.generate_response(f"""
            You are a City of Boston research assistant that is given context about a user's query
            and performs targeted web searches on topics related to city government to gather information that will
            help downstream agents respond to the query. 
            
            Here is a user query:
            {query}
            
            Search the web for relevant information and return the links.
        """)

        # Gather any annotations
        annotations = []
        for output in response.output:
            if output.type == 'message':
                for content in output.content:
                    if content.type == 'output_text' and content.annotations:
                        annotations += content.annotations

        return (response.output_text, annotations)