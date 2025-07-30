import os, asyncio, datetime
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from utils.ai_session import Session  # noqa: E402  (after sys.path tweak)

BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
if not (BOT_TOKEN and APP_TOKEN):
    raise RuntimeError(
        "❌  SLACK_BOT_TOKEN and/or SLACK_APP_TOKEN env-vars are missing."
    )

app = AsyncApp(token=BOT_TOKEN)

_sessions: dict[str, Session] = {}

test_session  = Session('testsession', 'shresth', exp=1234567890)

def _conversation_id(body: dict) -> str:
    """Return a stable id for each DM or thread in a channel."""
    # DM → channel id is the convo, channel message → use thread_ts or event ts
    ev = body["event"]
    
    if ev.get("channel_type") == "im":
        return ev["channel"]          # DM id (D123…)
    return ev.get("thread_ts") or ev["ts"]  # thread or root message ts

def _get_session(user_id: str, conv_id: str) -> Session:
    return test_session
    key = f"{user_id}::{conv_id}"
    sess = _sessions.get(key)
    now = int(datetime.datetime.utcnow().timestamp())
    if sess is None or getattr(sess, "exp", 0) < now:
        sess = Session(key, user_id, exp=now + 2 * 360000)  # 2-hour TTL
        _sessions[key] = sess
    return sess

async def _ask(session: Session, text: str) -> str:
    chunks = []
    output = session.run_query(text, sync=True)
    async for chunk in output:
        chunks.append(chunk)
    return "".join(chunks).strip() or "_(no answer)_"

# ---------------------------------------------------------------------
#  SLACK LISTENERS
# ---------------------------------------------------------------------
@app.event("app_mention")
async def handle_mention(body, say, logger):
    uid = body["event"]["user"]
    conv_id = _conversation_id(body)
    text = body["event"]["text"].split(None, 1)[1] if " " in body["event"]["text"] else ""
    logger.info(f"[mention] {uid} → {text!r}")
    sess = _get_session(uid, conv_id)
    reply = await _ask(sess, text)
    await say(reply, thread_ts=conv_id)

@app.event("message")
async def handle_dm(body, say, logger):
    if body["event"].get("channel_type") != "im":
        return
    uid = body["event"]["user"]
    conv_id = _conversation_id(body)
    print(f'conv_id, uid : {conv_id}, {uid}')
    logger.info(f'conv_id, uid : {conv_id}, {uid}')
    text = body["event"]["text"]
    logger.info(f"[dm] {uid} → {text!r}")
    sess = _get_session(uid, conv_id)
    reply = await _ask(sess, text)
    await say(reply)

async def main():
    handler = AsyncSocketModeHandler(app, APP_TOKEN)
    await handler.start_async()  # blocks forever

if __name__ == "__main__":
    asyncio.run(main())
