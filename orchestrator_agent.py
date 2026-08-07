from openai import OpenAI
from agents import Agent, Runner, WebSearchTool, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from rag_agent.rag_agent import rag_agent, pop_sources
from agents.guardrail import input_guardrail
import json, time 
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
            Synthesize a succint and helpful response (at most 2 paragraphs) to the user's query
            using only context from the tool results. 
            If there is conflicting context from different tool results, prioritize the results from the rag tool."""
        )

        self.guardrail_agent = Agent(
            name='guardrail',
            model=self.model,
            instructions="""
                Check if the user's query is related to Boston city government. 
                If it is unrelated, return `True`. Otherwise, return `False`
            """,
            output_type=bool,
        )

        @input_guardrail(run_in_parallel=False)
        async def relevance_guardrail(
            ctx: RunContextWrapper[None], 
            agent: Agent, input: str | list[TResponseInputItem]
        ) -> GuardrailFunctionOutput:
            result = await Runner.run(self.guardrail_agent, input, context=ctx.context)
            return GuardrailFunctionOutput(
                output_info=result.final_output, 
                tripwire_triggered=result.final_output,
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
                    tool_description='Can search the Internet for relevant City of Boston information'
                )
            ],
            input_guardrails=[relevance_guardrail],
            instructions="""
                You are a City of Boston research assistant that responds to citizens' queries.
                Use the tools you are given, at your discretion, to gather information that is relevant to a query.
                If you find sufficient information about a public notice/event using the rag tool, it is not necessary to also use the web_search tool. 
                The information you gather will be passed to another agent, which will deliver a final answer 
                to the user based on the information you provide.
            """
        )



    async def answer(self, user_query: str) -> tuple[str, list[dict], dict]:
        t0 = time.perf_counter()
        trace_info = {"question": user_query, "tool_calls": []}
        calls_by_id = {}

        try:
            result = await Runner.run(self.orchestrator_agent, user_query)

            for item in result.new_items:
                if item.type == "tool_call_item":
                    raw = item.raw_item
                    args = getattr(raw, "arguments", None)
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        pass  # keep the raw string if the model emitted something odd
                    call = {
                        "name": getattr(raw, "name", type(raw).__name__),
                        "input": args,
                        "output": None,
                }
                    trace_info["tool_calls"].append(call)
                    call_id = getattr(raw, "call_id", None)
                    if call_id:
                        calls_by_id[call_id] = call

                elif item.type == "tool_call_output_item":
                    call_id = (item.raw_item or {}).get("call_id")
                    if call_id in calls_by_id:
                        calls_by_id[call_id]["output"] = item.output
                    else:
                        trace_info["tool_calls"].append({"name": None, "input": None, "output": item.output})

            sources = pop_sources()
            final_answer = (await Runner.run(self.answer_agent, result.final_output)).final_output

        except InputGuardrailTripwireTriggered:
            final_answer = "Sorry, that doesn't seem to be related to the government of the City of Boston."
            sources = []
            trace_info["guardrail_tripped"] = True

        trace_info["tools_called"] = [c["name"] for c in trace_info["tool_calls"]]
        trace_info["sources"] = sources
        trace_info["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return final_answer, sources, trace_info