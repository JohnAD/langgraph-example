from __future__ import annotations

import os
from typing import Literal, Annotated, TypedDict

from braintrust import init_logger
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from src.agent.weathergov import get_weather_forecast
from src.agent.prompts import get_leading_prompts
from src.agent.weathergov import summarize_forecasts

init_logger(project="My Project", api_key=os.environ.get("BRAINTRUST_API_KEY"))
handler = BraintrustCallbackHandler()
set_global_handler(handler)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def get_weather(gridId: str, x: str, y: str) -> str:
    """Get the weather for a given location expressed as gridId, x, y."""
    forecasts = get_weather_forecast(gridId, x, y)
    return summarize_forecasts(forecasts)


TOOLS = "tools"
tools = [get_weather]
tool_node = ToolNode(tools)

llm = ChatOpenAI(
    api_key=SecretStr(os.environ['TA_OPENAI_KEY']),
    model="gpt-4o"
)

llm_with_tools = llm.bind_tools(tools)

ASSISTANT = "assistant"


async def assistant(state: State):
    all_messages = get_leading_prompts() + list(state["messages"])
    return {
        "messages": [await llm_with_tools.ainvoke(all_messages)]
    }


def should_continue(state: State) -> Literal['tools', '__end__']:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Define the graph
graph = StateGraph(State)
graph.add_node(TOOLS, tool_node)
graph.add_node(ASSISTANT, assistant)
graph.add_edge(TOOLS, ASSISTANT)
graph.add_conditional_edges(ASSISTANT, should_continue)
graph.set_entry_point(ASSISTANT)

graph = graph.compile()
