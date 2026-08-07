from openai.types.responses import Response
from orchestrator_agent import Agent


class AnswerAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def answer(self,query:str,research:list): 
        '''
        research: [(info, citations, agent)]
        '''
        # Format the research
        formatted_research = "\n".join([f"Agent: {agent} - Research: {info} - Citations:{citations}" for info,citations, agent in research])
        response = self.generate_response(f"""
            You are an assistant that is crafting a succint response with a professional but helpful 
            tone about a user's query based only on the provided context. If there is information from 
            multiple agents in the context, prioritize the information from the RAGAgent. Be sure to 
            present the Citation information so the user can clearly trace where the info is coming from.
            Query: 
            {query}
            Context:
            {formatted_research}
            
        """)
        return (response.output_text, [])



    