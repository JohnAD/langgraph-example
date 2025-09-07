import asyncio
import os
from typing import Dict

from braintrust import init_logger, traced, wrap_openai
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr


init_logger(project="My Project", api_key="sk-dLnw91yQDDTutA9IhW0ts0TaklYZVDiTxZ9ku9CBLD18ruaY")

# handler = BraintrustCallbackHandler()
# set_global_handler(handler)

# Use any LangChain supported model here
model = wrap_openai(ChatOpenAI(
    api_key=SecretStr(os.environ['TA_OPENAI_KEY']),
    model="gpt-4o-mini"
))

@traced
def say_hello(state: Dict[str, str]):
    response = model.invoke("Say hello")
    return response.content

@traced
def say_bye(state: Dict[str, str]):
    print("From the 'sayBye' node: Bye world!")
    return "Bye"

# Create the state graph
workflow = (
    StateGraph(state_schema=Dict[str, str])
    .add_node("sayHello", say_hello)
    .add_node("sayBye", say_bye)
    .add_edge(START, "sayHello")
    .add_edge("sayHello", "sayBye")
    .add_edge("sayBye", END)
)

graph = workflow.compile()
# asyncio.run(graph.ainvoke({"foo": "bar"}))
