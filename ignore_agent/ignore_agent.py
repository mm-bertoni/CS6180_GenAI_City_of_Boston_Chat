from orchestrator_agent import SubAgent

class IgnoreAgent(SubAgent):
    """
    Subagent that ignores the user's query. 
    For situations where the user's query is out of the intended scope of this tool.
    """
    def answer(self, query):
        return ("User's question is out of this tool's scope.", [])