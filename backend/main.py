from __future__ import annotations

import os
import logging
from typing import Optional
from collections import defaultdict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel


# Load environment variables from a local .env file (POC convenience).
load_dotenv()


# Configure logging
logger = logging.getLogger("chat_handoff_poc")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Optional OpenAI configuration for LLM-powered bot responses.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_openai_client: Optional[OpenAI] = None
if os.getenv("OPENAI_API_KEY"):
    _openai_client = OpenAI()

# Chatwoot configuration
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "").rstrip("/")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID", "")
CHATWOOT_INBOX_ID = os.getenv("CHATWOOT_INBOX_ID", "")

logger.info("=== Chatwoot Configuration ===")
logger.info("BASE_URL: %s", CHATWOOT_BASE_URL or "(not set)")
logger.info("ACCOUNT_ID: %s", CHATWOOT_ACCOUNT_ID or "(not set)")
logger.info("INBOX_ID: %s", CHATWOOT_INBOX_ID or "(not set)")
logger.info("API_TOKEN: %s", "***" if CHATWOOT_API_TOKEN else "(not set)")
logger.info("==============================")

app = FastAPI(title="Real-Time Chatbot POC")

# Global in-memory state (POC only – no persistence).
_human_mode: bool = False
_user_conversations: dict[str, int] = {}  # user_id -> chatwoot_conversation_id
_conversation_to_user: dict[int, str] = {}  # conversation_id -> user_id
_pending_agent_messages: dict[str, list[dict]] = defaultdict(list)  # user_id -> [messages]
# Track users for whom the bot has asked, "Do you want to talk to a human?"
_awaiting_handoff_confirmation: set[str] = set()
# Full transcript for each user, in order: [{"role": "user|assistant|agent|system", "content": str}, ...]
_conversation_history: dict[str, list[dict]] = defaultdict(list)

DEFAULT_USER_ID = "web-user"  # Single-user POC


# Allow browser access from the same origin (and others for POC simplicity).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HumanToggleRequest(BaseModel):
    """
    Payload for explicitly switching a user between bot and human.

    In the current POC we only have a single user (`DEFAULT_USER_ID`),
    but we keep `user_id` here so this endpoint matches the UI flow:
    "this specific user requested to talk to a human".
    """

    enabled: bool
    user_id: str = DEFAULT_USER_ID


class UserMessageRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    message: str


class UserOnlyRequest(BaseModel):
    """Simple payload used for endpoints that only need a user_id."""

    user_id: str = DEFAULT_USER_ID


def chatwoot_headers() -> dict:
    """Headers for Chatwoot API requests."""
    return {
        "api_access_token": CHATWOOT_API_TOKEN,
        "Content-Type": "application/json",
    }


def get_or_create_chatwoot_conversation(user_id: str) -> int:
    """
    Create or retrieve a Chatwoot conversation for the given user.
    
    Steps:
    1. Create a contact (or find existing)
    2. Create a conversation in the configured inbox
    3. Cache the conversation ID
    
    Returns the conversation_id.
    """
    if user_id in _user_conversations:
        logger.info("Using cached conversation for user=%s: conversation_id=%s", user_id, _user_conversations[user_id])
        return _user_conversations[user_id]

    logger.info("Creating new Chatwoot conversation for user=%s", user_id)

    # Step 1: Search for existing contact first
    search_contact_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts/search"
    contact_id = None
    
    logger.info("GET %s?q=%s", search_contact_url, user_id)
    try:
        r_search = requests.get(
            search_contact_url,
            params={"q": user_id},
            headers=chatwoot_headers(),
            timeout=10,
        )
        
        if r_search.ok:
            search_results = r_search.json()
            logger.debug("Contact search results: %s", search_results)
            
            # Check if contact exists with matching identifier
            payload = search_results.get("payload", [])
            for contact in payload:
                if contact.get("identifier") == user_id:
                    contact_id = contact.get("id")
                    logger.info("✅ Found existing contact: contact_id=%s", contact_id)
                    break
    except Exception as e:
        logger.warning("Contact search failed (will try create): %s", e)
    
    # Step 2: If contact doesn't exist, create it
    if not contact_id:
        create_contact_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts"
        contact_payload = {
            "name": user_id,
            "identifier": user_id,
        }
        
        logger.info("POST %s", create_contact_url)
        logger.debug("Contact payload: %s", contact_payload)
        
        try:
            r_contact = requests.post(
                create_contact_url,
                json=contact_payload,
                headers=chatwoot_headers(),
                timeout=10,
            )
            
            # Log response for debugging
            logger.debug("Contact API status: %s", r_contact.status_code)
            logger.debug("Contact API response body: %s", r_contact.text)
            
            if not r_contact.ok:
                logger.error("❌ Chatwoot contact API error %s: %s", r_contact.status_code, r_contact.text)
            
            r_contact.raise_for_status()
            contact_data = r_contact.json()
            
            # Handle different response formats
            if "payload" in contact_data and "contact" in contact_data["payload"]:
                contact_id = contact_data["payload"]["contact"]["id"]
            elif "id" in contact_data:
                contact_id = contact_data["id"]
            else:
                raise ValueError(f"Unexpected contact response format: {contact_data}")
                
            logger.info("✅ Contact created: contact_id=%s", contact_id)
            
        except requests.exceptions.RequestException as e:
            logger.exception("❌ Failed to create contact for user=%s: %s", user_id, e)
            raise
        except (KeyError, ValueError) as e:
            logger.exception("❌ Failed to parse contact response: %s", e)
            raise

    # Step 2: Create conversation
    create_convo_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations"
    convo_payload = {
        "source_id": user_id,
        "inbox_id": int(CHATWOOT_INBOX_ID),
        "contact_id": contact_id,
    }
    
    logger.info("POST %s", create_convo_url)
    logger.debug("Conversation payload: %s", convo_payload)
    
    try:
        r_convo = requests.post(
            create_convo_url,
            json=convo_payload,
            headers=chatwoot_headers(),
            timeout=10,
        )
        
        # Log response for debugging
        logger.debug("Conversation API status: %s", r_convo.status_code)
        logger.debug("Conversation API response body: %s", r_convo.text)
        
        if not r_convo.ok:
            logger.error("❌ Chatwoot conversation API error %s: %s", r_convo.status_code, r_convo.text)
        
        r_convo.raise_for_status()
        convo_data = r_convo.json()
        
        # Handle different response formats
        if "id" in convo_data:
            conversation_id = int(convo_data["id"])
        elif "payload" in convo_data and "id" in convo_data["payload"]:
            conversation_id = int(convo_data["payload"]["id"])
        else:
            raise ValueError(f"Unexpected conversation response format: {convo_data}")
            
        logger.info("✅ Conversation created: conversation_id=%s", conversation_id)
        
        # Cache the mapping
        _user_conversations[user_id] = conversation_id
        _conversation_to_user[conversation_id] = user_id
        
        return conversation_id
        
    except requests.exceptions.RequestException as e:
        logger.exception("❌ Failed to create conversation for user=%s: %s", user_id, e)
        raise
    except (KeyError, ValueError) as e:
        logger.exception("❌ Failed to parse conversation response: %s", e)
        raise


def send_message_to_chatwoot(conversation_id: int, message: str) -> dict:
    """
    Send a message to Chatwoot as 'incoming' (from the customer/user).
    
    This makes the message appear in the Chatwoot inbox for agents to see.
    
    Note: message_type should be INTEGER:
    - 0 = incoming (from contact/customer)
    - 1 = outgoing (from agent/bot)
    """
    url = (
        f"{CHATWOOT_BASE_URL}/api/v1/accounts/"
        f"{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    )
    payload = {
        "content": message,
        "message_type": 0,  # 0 = incoming (from customer)
        "private": False,   # Make it visible to customer
    }
    
    logger.info("POST %s", url)
    logger.debug("Message payload: %s", payload)
    
    try:
        r = requests.post(url, json=payload, headers=chatwoot_headers(), timeout=10)
        
        # Log response regardless of status for debugging
        logger.debug("Chatwoot response status: %s", r.status_code)
        logger.debug("Chatwoot response body: %s", r.text)
        
        if not r.ok:
            # Log the error details before raising
            logger.error("❌ Chatwoot API error %s: %s", r.status_code, r.text)
        
        r.raise_for_status()
        msg_data = r.json()
        logger.info("✅ Message sent to Chatwoot: message_id=%s", msg_data.get("id"))
        logger.debug("Message API response: %s", msg_data)
        return msg_data
    except requests.exceptions.RequestException as e:
        logger.exception("❌ Failed to send message to Chatwoot conversation=%s: %s", conversation_id, e)
        raise


def generate_bot_response(user_text: str) -> str:
    """
    Generate a response for the user.

    - If `OPENAI_API_KEY` is configured, this will call the OpenAI Chat
      Completions API to generate a real LLM answer.
    - Otherwise, it falls back to a simple mock response so the POC still
      runs without any external services.
    """
    cleaned = (user_text or "").strip()
    if not cleaned:
        cleaned = "Say hello to the user and ask how you can help."

    # If no OpenAI client is configured, use a local mock response.
    if _openai_client is None:
        logger.info("OPENAI_API_KEY not set; using mock AI response.")
        return f"Hello! How can I assist you today?"

    try:
        logger.info("Calling OpenAI model '%s' for bot response.", OPENAI_MODEL)
        completion = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise, friendly support chatbot in a proof-of-concept demo.",
                },
                {"role": "user", "content": cleaned},
            ],
            max_tokens=200,
        )
        message = completion.choices[0].message.content or ""
        logger.debug("OpenAI raw completion: %s", completion)
        return message.strip()
    except Exception as exc:
        logger.exception("Error while calling OpenAI API: %s", exc)
        return f"Hello! How can I assist you today?"


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def user_seems_to_request_human(text: str) -> bool:
    """
    Heuristic check: does the user expression look like
    they want to talk to a human?
    """
    normalized = _normalize_text(text)
    if not normalized:
        return False

    keywords = [
        "human",
        "agent",
        "real person",
        "live person",
        "live agent",
        "customer care",
        "support team",
        "talk to someone",
        "speak to someone",
        "speak to a person",
        "talk to a person",
    ]
    for kw in keywords:
        if kw in normalized:
            return True
    return False


def user_confirms_handoff(text: str) -> bool:
    """
    Detect simple yes/affirmative confirmations.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return False

    affirmatives = {
        "yes",
        "y",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "please",
        "please do",
        "connect me",
    }
    return normalized in affirmatives or normalized.startswith("yes ")


def user_declines_handoff(text: str) -> bool:
    """
    Detect simple no/negative confirmations.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return False

    negatives = {"no", "n", "nope", "not now", "never mind"}
    return normalized in negatives


def add_history_entry(user_id: str, role: str, content: str) -> None:
    """
    Append a single conversational turn to the in-memory transcript.

    role is one of: "user", "assistant", "agent", "system".
    """
    cleaned = (content or "").strip()
    if not cleaned:
        return
    _conversation_history[user_id].append(
        {
            "role": role,
            "content": cleaned,
        }
    )


def build_conversation_summary(user_id: str) -> str:
    """
    Build a plain-text transcript for Chatwoot so the human agent
    can see the full context as the first message.
    """
    history = _conversation_history.get(user_id) or []
    if not history:
        return "User started a conversation, but there is no previous transcript."

    role_labels = {
        "user": "User",
        "assistant": "Bot",
        "agent": "Human agent",
        "system": "System",
    }

    lines: list[str] = []
    for item in history:
        role = role_labels.get(item.get("role", "user"), "User")
        content = item.get("content", "")
        if not content:
            continue
        lines.append(f"{role}: {content}")

    header = "Conversation so far (share with human agent):"
    return header + "\n" + "\n".join(lines)


@app.post("/toggle-human")
async def toggle_human_mode(payload: HumanToggleRequest, request: Request):
    """
    REST endpoint to explicitly turn human mode on/off.

    Body: {"enabled": true/false}
    """
    client_host = request.client.host if request.client else "unknown"
    user_id = payload.user_id or DEFAULT_USER_ID
    logger.info(
        "Human mode toggle requested from %s for user=%s: enabled=%s",
        client_host,
        user_id,
        payload.enabled,
    )
    global _human_mode
    _human_mode = bool(payload.enabled)
    logger.info("Human mode is now %s", _human_mode)
    
    # If turning ON human mode, pre-create the conversation
    if _human_mode:
        try:
            conversation_id = get_or_create_chatwoot_conversation(DEFAULT_USER_ID)
            logger.info("✅ Conversation ready for human handoff: conversation_id=%s", conversation_id)
        except Exception as e:
            logger.error("❌ Failed to create conversation during human mode toggle: %s", e)
    
    return {"human_mode": _human_mode}


@app.get("/human-status")
async def human_status():
    """
    Simple endpoint to inspect current human mode.
    Useful for debugging and demos.
    """
    return {
        "human_mode": _human_mode,
    }


@app.post("/connect-human-now")
async def connect_human_now(payload: UserOnlyRequest, request: Request):
    """
    Explicitly switch this user to human mode and push the full transcript
    into Chatwoot as the first message.

    This is used by the header hamburger menu for a direct handoff,
    without requiring a yes/no free-text confirmation in the chat.
    """
    global _human_mode

    client_host = request.client.host if request.client else "unknown"
    user_id = payload.user_id or DEFAULT_USER_ID
    logger.info("Direct human connect requested from %s for user=%s", client_host, user_id)

    try:
        conversation_id = get_or_create_chatwoot_conversation(user_id)
        transcript = build_conversation_summary(user_id)
        send_message_to_chatwoot(conversation_id, transcript)

        _human_mode = True
        logger.info(
            "✅ Direct human connect complete for user=%s, conversation_id=%s",
            user_id,
            conversation_id,
        )

        return {
            "status": "forwarded",
            "route": "human",
            "conversation_id": conversation_id,
            "message": "Conversation history sent to human agent in Chatwoot.",
            "human_mode": _human_mode,
        }
    except Exception as exc:
        logger.exception("❌ Direct human connect failed for user=%s: %s", user_id, exc)
        return {
            "status": "error",
            "message": "Failed to connect you to a human agent.",
            "human_mode": _human_mode,
        }


@app.post("/clear-chat")
async def clear_chat(payload: UserOnlyRequest, request: Request):
    """
    Clear the in-memory transcript and any pending state for this user.

    This is triggered from the UI hamburger menu "Clear chat" option.
    It also resets human mode back to bot.
    """
    global _human_mode

    client_host = request.client.host if request.client else "unknown"
    user_id = payload.user_id or DEFAULT_USER_ID
    logger.info("Clear chat requested from %s for user=%s", client_host, user_id)

    # Reset per-user state
    _conversation_history.pop(user_id, None)
    _pending_agent_messages.pop(user_id, None)
    _awaiting_handoff_confirmation.discard(user_id)

    # Go back to bot mode for a fresh start
    _human_mode = False

    logger.info("Chat cleared for user=%s; human_mode reset to False.", user_id)
    return {
        "status": "cleared",
        "human_mode": _human_mode,
    }


@app.post("/message")
async def receive_message(data: UserMessageRequest, request: Request):
    """
    Receive a message from the user.
    
    - If human mode is OFF: generate bot reply and return it
    - If human mode is ON: send to Chatwoot and return confirmation
    """
    user_id = data.user_id or DEFAULT_USER_ID
    message = data.message
    
    client_host = request.client.host if request.client else "unknown"
    logger.info("Message from %s (user=%s): %s", client_host, user_id, message[:50])
    
    if not message.strip():
        return {"status": "error", "message": "Empty message"}

    # Always record the user's message in the transcript.
    add_history_entry(user_id, "user", message)

    global _human_mode

    if _human_mode:
        # HUMAN MODE: Send to Chatwoot
        logger.info("🙋 HUMAN MODE: Forwarding message to Chatwoot")
        
        try:
            conversation_id = get_or_create_chatwoot_conversation(user_id)
            send_message_to_chatwoot(conversation_id, message)
            
            return {
                "status": "forwarded",
                "route": "human",
                "conversation_id": conversation_id,
                "message": "Message sent to human agent in Chatwoot"
            }
        except Exception as e:
            logger.exception("❌ Failed to forward message to Chatwoot: %s", e)
            return {
                "status": "error",
                "message": f"Failed to send to Chatwoot: {str(e)}"
            }
    # BOT MODE (no active human handoff yet)
    logger.info("🤖 BOT MODE: Processing message")

    # If we've already asked this user whether they want a human, interpret
    # this message as their confirmation (yes/no).
    if user_id in _awaiting_handoff_confirmation:
        if user_confirms_handoff(message):
            logger.info("User %s confirmed human handoff; enabling human mode.", user_id)
            _awaiting_handoff_confirmation.discard(user_id)
            _human_mode = True

            try:
                conversation_id = get_or_create_chatwoot_conversation(user_id)
                logger.info("✅ Conversation ready after confirmation: conversation_id=%s", conversation_id)

                # Build and send the full transcript as the very first message
                # into Chatwoot so the human agent sees all prior context.
                transcript = build_conversation_summary(user_id)
                send_message_to_chatwoot(conversation_id, transcript)
            except Exception as exc:
                logger.exception("❌ Failed to prepare Chatwoot conversation after confirmation: %s", exc)
                # Even if Chatwoot fails, stay in bot mode so the user is not stuck.
                _human_mode = False
                return {
                    "status": "error",
                    "route": "bot",
                    "message": "Tried to connect you to a human agent, but something went wrong.",
                    "human_mode": _human_mode,
                }

            return {
                "status": "forwarded",
                "route": "human",
                "conversation_id": conversation_id,
                "message": "Conversation history sent to human agent in Chatwoot.",
                "human_mode": _human_mode,
            }

        if user_declines_handoff(message):
            logger.info("User %s declined human handoff; staying in bot mode.", user_id)
            _awaiting_handoff_confirmation.discard(user_id)
            reply = "No problem — I’ll keep helping you here as the bot. How else can I assist you?"
            add_history_entry(user_id, "assistant", reply)
            return {
                "status": "bot",
                "route": "bot",
                "reply": reply,
                "human_mode": _human_mode,
            }

        # If the user replied with something ambiguous, clear the pending
        # confirmation and continue in bot mode with a normal reply.
        logger.info(
            "User %s sent ambiguous response while awaiting confirmation; "
            "clearing confirmation flag and continuing in bot mode.",
            user_id,
        )
        _awaiting_handoff_confirmation.discard(user_id)

    # If the user seems to explicitly ask for a human, ask for confirmation
    # in the conversational flow instead of toggling immediately.
    if user_seems_to_request_human(message):
        logger.info("User %s appears to be requesting a human agent; asking for confirmation.", user_id)
        _awaiting_handoff_confirmation.add(user_id)
        reply = (
            "It sounds like you’d like to talk to a human agent. "
            "If that’s right, please reply with “yes” and I’ll connect you. "
            "If not, you can reply “no” to keep chatting with me."
        )
        add_history_entry(user_id, "assistant", reply)
        return {
            "status": "bot",
            "route": "bot",
            "reply": reply,
            "human_mode": _human_mode,
            # Hint to the frontend that it can render quick-action buttons
            # instead of relying solely on free-text yes/no replies.
            "human_handoff_confirmation": True,
        }

    # Default: pure bot reply.
    logger.info("🤖 BOT MODE: Generating standard bot reply")
    reply = generate_bot_response(message)
    logger.info("Bot reply: %s", reply[:50])
    add_history_entry(user_id, "assistant", reply)

    return {
        "status": "bot",
        "route": "bot",
        "reply": reply,
        "human_mode": _human_mode,
    }


@app.post("/webhook/chatwoot")
async def chatwoot_webhook(request: Request):
    """
    Webhook endpoint to receive events from Chatwoot.
    
    Configure this URL in Chatwoot:
    Settings → Integrations → Webhooks → Add Webhook
    URL: https://your-domain.com/webhook/chatwoot
    Events: message_created
    
    This endpoint receives agent replies and makes them available to the frontend.
    """
    client_host = request.client.host if request.client else "unknown"
    
    try:
        payload = await request.json()
    except Exception as e:
        logger.error("❌ Failed to parse webhook payload: %s", e)
        return {"status": "error", "message": "Invalid JSON"}
    
    logger.info("📨 Webhook from %s: event=%s", client_host, payload.get("event"))
    logger.info("🔍 FULL WEBHOOK PAYLOAD: %s", payload)  # Log entire payload

    event = payload.get("event")
    if event != "message_created":
        logger.info("Ignoring non-message_created event: %s", event)
        return {"status": "ignored"}

    # ----- Normalize payload shape -------------------------------------------------
    # Chatwoot webhook docs show both of these shapes:
    # 1) Top-level fields:
    #    { event, id, content, message_type, conversation, ... }
    # 2) Nested message object:
    #    { event, message: { content, message_type, ... }, conversation: {...}, ... }
    #
    # To be robust, treat `message` (if present) as the source of truth,
    # otherwise fall back to top-level fields.
    # ------------------------------------------------------------------------------
    raw_message = payload.get("message") or payload

    message_type = raw_message.get("message_type")  # 0 = incoming (customer), 1 = outgoing (agent)
    content = raw_message.get("content", "")

    conversation = payload.get("conversation") or raw_message.get("conversation") or {}
    conversation_id = conversation.get("id")

    logger.info(
        "Message details (normalised): type=%s, conversation_id=%s, content=%s",
        message_type,
        conversation_id,
        content[:30] if content else "",
    )

    # Only process AGENT replies.
    # Chatwoot may send:
    #   - integer 1 for agent message_type
    #   - string "outgoing" in some payloads
    # We'll treat both as agent replies.
    if message_type == 1 or message_type == "outgoing":
        logger.info("👤 Agent reply detected")
        
        # Find the user for this conversation
        user_id = _conversation_to_user.get(conversation_id)
        if user_id:
            logger.info("✅ Queueing agent reply for user=%s: %s", user_id, content[:50] if content else content)
            add_history_entry(user_id, "agent", content or "")
            _pending_agent_messages[user_id].append({
                "content": content,
                "conversation_id": conversation_id,
            })
        else:
            logger.warning("⚠️  No user mapping for conversation_id=%s", conversation_id)
    else:
        logger.info("Ignoring non-agent message (type=%s)", message_type)
    
    return {"status": "ok"}


@app.get("/poll-human-messages")
async def poll_human_messages(user_id: str = DEFAULT_USER_ID):
    """
    Poll endpoint for the frontend to check for new agent messages.
    
    Returns all pending messages and clears the queue.
    """
    messages = _pending_agent_messages.pop(user_id, [])
    if messages:
        logger.info("📬 Delivering %d agent messages to user=%s", len(messages), user_id)
    return {"messages": messages}


@app.get("/")
async def serve_index():
    """
    Serve the single-page frontend (plain HTML + vanilla JS).
    """
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
