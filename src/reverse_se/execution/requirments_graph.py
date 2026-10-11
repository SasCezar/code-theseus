from __future__ import annotations

from typing import TypedDict

from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    sys_prompt: str
    nodes: str
    edges: str
    response: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior software architect. You need only a dependency graph, and high level overview of a repository and you can quickly reverse engineer the original architecture. You identify architecture patterns, compoments, and write all your knowleadge in a C4 Design document. The document needs to seem like it was always existing, long before the implementation. So that the company looks like it is following the best and modern software development cycles. You only output a C4 Design document. You provide diagrams of each level, and high level descriptions. \n {out_document}",
        ),
        (
            "human",
            "Given the following graph :\n\nNODES:\n{nodes}\n\nEDGES:\n{edges}\n\n",
        ),
    ]
)


def call_llm(state: GraphState) -> GraphState:
    messages: list[BaseMessage] = prompt.format_messages(
        sys_prompt=state["sys_prompt"],
        nodes=state["nodes"],
        edges=state["edges"],
    )
    ai_msg: AIMessage = llm.invoke(messages)
    return {"response": ai_msg.content}


builder = StateGraph(GraphState)
builder.add_node("call_llm", call_llm)

builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

architecture_graph = builder.compile()
