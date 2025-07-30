import os
from utils.datetime_utils import get_current_time_with_offset
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from utils.llmUtils import getLLM, filter_messages
from utils.pr_utils import fetch_pr_diff, slice_diff, annotate_diff
from tools.tools_list import tools_list

tool_node = ToolNode(tools_list["PREVIEW"])
# --- BASE SYSTEM PROMPT ---
BASE_SYSTEM = """
    You are a senior developer reviewing a GitHub Pull Request (PR). Your goal is to help your teammate by pointing out specific improvements.

    ❌ Do NOT summarize the changes. Instead, focus on:
    1. Optimizing code for latency/performance
    2. Catching bugs or edge cases
    3. Improving readability and maintainability

    ✅ For each suggestion:
    - Include concrete code examples
    - Mention the relevant **filename**
    - Be specific and helpful, like you’re pairing with a junior dev

    📝 Tone:
    - Use friendly, informal language (a hint of Hinglish is okay!)
    - No unnecessary politeness, just solid, constructive feedback

    📎 Formatting rules:
    - No **bold**, *italic*, or special styling
    - Use plain text with clear breaks or emojis to improve readability in Slack
    - Optional: End with a short summary after all detailed suggestions
"""

# --- Chat Prompt ---
instruction_prompt = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM),
    ("placeholder", "{messages}")
])

llm = getLLM("openai", "PREVIEW")

# 2) bind the tools to *that* model
llm_with_tools = llm.bind_tools(tools_list["PREVIEW"])

# 3) finally add the instruction prompt on top
model_with_prompt = instruction_prompt | llm_with_tools


# --- Core invoke function for assistant node ---
async def invoke(state: dict, config: RunnableConfig):
    query = next(m.content for m in state["messages"][::-1] if m.type == "human")
    diff_raw = await fetch_pr_diff(query, token=os.getenv("GITHUB_PAT"))
    if not diff_raw:
        return {"messages": [("system","❌ Unable to fetch diff.")]}

    diff_cut = slice_diff(diff_raw, context=3)
    diff_trim = annotate_diff(diff_cut)
    # Clean messages and append diff
    state["messages"] = filter_messages(state["messages"])
    state["messages"].append(("system", f"Today is {get_current_time_with_offset(config).strftime('%d-%m-%Y %H:%M:%S')}. TimeZone configured for the logged in user in the Mobile Cloud is {config.get('configurable', {}).get('timezone', 'Asia/Kolkata')}, with an offset of {config.get('configurable', {}).get('timezone_offset', 330)} minutes."))
    state["messages"].append(("user",f"```Pull Request difference : \n{diff_trim}\n```"))
    result = await model_with_prompt.ainvoke(state, config=config)
    return {"messages": [result]}

# --- Conditional Edge Logic ---
def should_continue(state):
    last_message = state["messages"][-1]
    return "continue" if getattr(last_message, "tool_calls", None) else "end"

# --- LangGraph Setup ---
memory = MemorySaver()
graph = StateGraph(MessagesState)

graph.add_node("assistant", invoke)
graph.add_node("tools", tool_node)

graph.add_edge(START, "assistant")
graph.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "assistant")

# --- Compile Graph ---
preview = graph.compile(checkpointer=memory)
