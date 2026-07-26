from agents import WebSearchTool, Agent, Runner, Model
from asyncio import run

search_tool = WebSearchTool(
    search_context_size='medium', # takes value 'low', 'medium', 'high'
    external_web_access = True
)

agent = Agent(
    name='web_search', 
    instructions="""You are an assistant that is given context about a user's inquiry,
and formulates any web searches that may aid in responding to the inquiry.""",
    tools=[search_tool]
)

async def main():
    initial_context = input("Initial context for the web search agent: ")
    runner = Runner()
    result = await runner.run(agent, input=initial_context)
    print(result.final_output)

if __name__ == '__main__':
    run(main())