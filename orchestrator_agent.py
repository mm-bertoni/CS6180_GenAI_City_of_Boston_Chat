from typing import Tuple, List
from openai import OpenAI
from agents import Agent, Runner

type SubAgentResponse = Tuple[str, List[dict]]

class MultiAgent():
    """
    Orchestrates multiple agents to answer the user query.
    """

    def __init__(self):
        self.client = OpenAI()
        self.model = 'gpt-4o-mini'  # model used for orchestrator and all subagents

        # Subagents
        self.rag_agent = Agent(
            name='rag',
            model=self.model,
            #TODO: Adapt current rag agent here as an OpenAI Agents SDK agent
        )
        self.web_search_agent = Agent(
            name='web_search', 
            instructions=f"""
                You are a City of Boston research assistant that is given context about a user's query
                and performs targeted web searches on topics related to city government to gather information that will
                help downstream agents respond to the query. You search the web for information relevant to the user's query and return the links you find.
            """,
            model=self.model,
            output_type=SubAgentResponse
        )

        self.answer_agent = Agent(
            name='answer',
            model=self.model,
            handoff_description="""
                Assembles the final answer to the user. 
                Call this agent when you have gathered all the information you see fit
                using your tools and are ready for the user-facing answer to be synthesized.
            """
            #TODO: Adapt current answer agent here as an OpenAI Agents SDK agent
        )

        self.orchestrator_agent = Agent(
            name='orchestrator',
            model=self.model,
            tools=[
                self.rag_agent.as_tool(
                    tool_name='orchestrator',
                    tool_description='Can search through City of Boston public notices'
                ), 
                self.web_search_agent.as_tool(
                    tool_name='web_search',
                    tool_description='Can search the Internet for relevant information'
                )
            ],
            handoffs=[self.answer_agent],
            instructions="""
                You are a City of Boston research assistant that responds to citizens' queries.
                Use the tools you are given, at your discretion, to gather information that is relevant to a query.
                When you have gathered all the information you see fit, call upon the handoff agent,
                which will assemble and deliver a final answer to the user.
            """
        )

    async def answer(self, user_query: str) -> str:
        result = await Runner.run(self.orchestrator_agent, user_query)
        return result.final_output