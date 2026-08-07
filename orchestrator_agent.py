from openai import OpenAI
from agents import Agent, Runner, WebSearchTool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from rag_agent.rag_agent import rag_agent, pop_sources

class MultiAgent():
    """
    Orchestrates multiple agents to answer the user query.
    """

    def __init__(self):
        self.client = OpenAI()
        self.model = 'gpt-4o-mini'  # model used for orchestrator and all subagents

        # Subagents
        self.rag_agent = rag_agent
        self.web_search_agent = Agent(
            name='web_search', 
            instructions=f"""
                You are a City of Boston research assistant that is given context about a user's query
                and performs targeted web searches on topics related to city government to gather information that will
                help downstream agents respond to the query. You search the web for information relevant to the user's query and return the links you find.
            """,
            model='gpt-5-mini',  # filters= is unsupported on gpt-4o-mini
            tools=[WebSearchTool(
                filters={"allowed_domains":["boston.gov"]}
            )],
        )

        self.answer_agent = Agent(
            name='answer',
            model=self.model,
            handoff_description="""
                Assembles the final answer to the user. 
                Call this agent when you have gathered all the information you see fit
                using your tools and are ready for the user-facing answer to be synthesized.
            """,
            # Prefex is recommended by OpenAI for handoffs specifically
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}\n 
            Synthesize a succint, professional, but helpful response to the user's query
            using only context from the tool results. 
            If there is conflicting context from different tool results, prioritize the results from the rag tool."""
        )

        self.orchestrator_agent = Agent(
            name='orchestrator',
            model=self.model,
            tools=[
                self.rag_agent.as_tool(
                    tool_name='rag',
                    tool_description='Can search through City of Boston public notices'
                ), 
                self.web_search_agent.as_tool(
                    tool_name='web_search',
                    tool_description='Can search the Internet for relevant information'
                )
            ],
            instructions="""
                You are a City of Boston research assistant that responds to citizens' queries.
                Use the tools you are given, at your discretion, to gather information that is relevant to a query.
                The information you gather will be passed to another agent, which will deliver a final answer 
                to the user based on the information you provide.
            """
        )

    async def answer(self, user_query: str) -> tuple[str, list[dict]]:
        """
        Generate an answer for the given user query.
        Returns a tuple with (1) the generated answer and (2) a list of sources consulted.
        """
        orchestrator_result = (await Runner.run(self.orchestrator_agent, user_query)).final_output
        sources = pop_sources()
        final_answer = (await Runner.run(self.answer_agent, orchestrator_result)).final_output
        return final_answer, sources