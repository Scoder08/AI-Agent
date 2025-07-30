import traceback, logging, datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from constants import RECURSION_LIMIT
from utils.datetime_utils import timezone_to_offset
from agents.supervisor_agents import get_supervisor

logger = logging.getLogger(__name__)


class Session:
    """
    Keeps per-conversation memory and streams LangGraph (SUPERVISOR) output
    back to the caller chunk-by-chunk.
    """

    _app = None                # compiled LangGraph (singleton)

    def __init__(self, session_id: str, user: str, exp: int,
                 timezone: str = "Asia/Calcutta", max_turns: int = 100):
        self.__max_turns = max_turns
        self.__thread_id = f"{user}_{exp}"
        self.__session_id = session_id
        self.timezone = timezone

        # history starts with a single SystemMessage
        self.__state = {
            "messages": [
                SystemMessage(content=f"session_id={session_id}")
            ]
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    async def _ensure_graph_loaded(self):
        if Session._app is None:
            Session._app = await get_supervisor()

    def _trim_history(self) -> None:
        """Trim oldest turns so total messages ≤ max_turns * 2."""
        excess = len(self.__state["messages"]) - (self.__max_turns * 2)
        if excess > 0:
            self.__state["messages"] = self.__state["messages"][excess:]

    # ------------------------------------------------------------------ #
    # Public call
    # ------------------------------------------------------------------ #
    async def run_query(self, query: str, *, sync: bool = True):
        """
        Stream the supervisor‘s answer.  Yields chunks of text so callers
        (Slack bridge) can forward them as they arrive.

        Parameters
        ----------
        query : str
        sync  : bool   (kept to match earlier signature)
        """
        await self._ensure_graph_loaded()

        # ①  append new user message to stored history
        user_msg = HumanMessage(content=query)
        self.__state["messages"].append(user_msg)

        # ②  invoke LangGraph with full history
        state = {"messages": list(self.__state["messages"])}   # shallow copy
        config = {
            "configurable": {
                "thread_id": self.__thread_id,
                "session_id": self.__session_id,
                "timezone_offset": timezone_to_offset(self.timezone),
                "timezone": self.timezone,
                "sync": sync,
            },
            "recursion_limit": RECURSION_LIMIT,
        }

        assistant_chunks: list[str] = []
        prev_agent = ""
        try:
            # yield "```"                         # opening fence for Slack

            async for event in Session._app.astream_events(
                state, config, version="v2"
            ):
                kind, name = event["event"], event["name"]

                if kind == "on_chat_model_stream" and name == "SUPERVISOR":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        if name != prev_agent:
                            # yield f"<<< agent = {name} >>>"
                            prev_agent = name
                        assistant_chunks.append(chunk)
                        yield chunk

                elif kind == "on_chat_model_end" and name == "DEE":
                    content = event["data"]["output"].content
                    if content and content.startswith("/") and content != "/":
                        yield f"<<< link = {content} >>>"

                elif kind == "on_tool_start":
                    logger.info(f"→ {name} start {event['data'].get('input')}")
                elif kind == "on_tool_end":
                    logger.info(f"← {name} end {event['data'].get('output')}")

            # ③  store assistant reply into history
            assistant_text = "".join(assistant_chunks).strip()
            if assistant_text:
                self.__state["messages"].append(
                    AIMessage(content=assistant_text)
                )

            # ④  keep history bounded
            self._trim_history()

            # yield "```"                        # closing fence

        except Exception:
            logger.error(traceback.format_exc())
            yield "<<< agent = SUPERVISOR >>>"
            yield "Failed to answer the query. Please contact support."
            yield "<<< status = END >>>"
