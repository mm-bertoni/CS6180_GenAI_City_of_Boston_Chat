from agents import WebSearchTool, Agent, Runner, Model
from asyncio import run

class WebSearchAgent():
    def __init__(self):

        self._search_tool = WebSearchTool(
            search_context_size='medium', # takes value 'low', 'medium', 'high'
            external_web_access = True
        )
        self._agent = Agent(
            name='web_search', 
            instructions="""You are an assistant that is given context about a user's inquiry,
        and formulates any web searches that may aid in responding to the inquiry.""",
            tools=[self._search_tool]
        )

    async def answer(self, question: str) -> str:
        runner = Runner()
        result = await runner.run(self._agent, input=question)
        return result.final_output


# For running interactively:
async def main():
    interactive_question = input("Initial context for the web search agent: ")
    web_search_agent = WebSearchAgent()
    interactive_answer = await web_search_agent.answer(interactive_question)
    print(interactive_answer)

if __name__ == '__main__':
    run(main())