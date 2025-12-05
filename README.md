## Real-Time Chatbot POC (FastAPI + Webhooks)

sample env:
OPENAI_API_KEY=" "
# Optional: override default model
OPENAI_MODEL=gpt-4o-mini


CHATWOOT_BASE_URL=
CHATWOOT_API_TOKEN=
CHATWOOT_ACCOUNT_ID=
CHATWOOT_INBOX_ID=



This project is a small **proof of concept** for a chatbot system with **bot ↔ human handoff**, using a
**Chatwoot-style webhook** instead of WebSockets.

- **Backend**: FastAPI (Python) + HTTP webhooks  
- **Frontend**: Single-page HTML + vanilla JavaScript (no UI frameworks)  
- **State**: In-memory only (no database, POC-ready)  

It demonstrates routing user messages either to a **mock AI bot** or to a **connected human agent** in real time.

---

### Project Structure

- **`backend/`**
  - `main.py` – FastAPI application with:
    - Webhook endpoint for Chatwoot-style events: `POST /webhook/chatwoot`
    - REST endpoint to toggle human mode: `POST /toggle-human`
    - Optional status endpoint: `GET /human-status`
  - `requirements.txt` – Python dependencies
- **`frontend/`**
  - `index.html` – Single-page UI (chat window, Human toggle)

Run everything from the project root (where this `README.md` lives).

---

### Prerequisites

- Python **3.9+**
- `pip` available on your PATH

---

### Setup & Installation

From the project root (`D:\chatwood_poc` or equivalent):

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt
```

---

### How to Run

From inside the activated virtual environment (still in `backend/`):

```bash
uvicorn backend.main:app --reload
```

Then open the frontend in your browser:

- Navigate to: `http://localhost:8000/`

You should see the **User Chat** panel and the **Agent Console** (for demo purposes) on a single page.

---

### Core Features

- **Chatwoot-style webhook** endpoint at `POST /webhook/chatwoot`
- **Global “Human Mode” toggle** (bot vs human routing) via:
  - REST: `POST /toggle-human` with JSON body `{ "enabled": true | false }`
  - UI toggle button labeled **“Human”**
- **In-memory state**:
  - Global `human_mode` flag (true/false)
- **Simple bot behavior**:
  - When **Human Mode is OFF**, the system responds with a **mock AI** (or OpenAI) reply.
- **Human handoff behavior**:
  - When **Human Mode is ON**, the webhook simply acknowledges the event so a human
    agent platform (such as Chatwoot) can handle the reply itself.

---

### How the Architecture Works

- **Chatwoot-style webhook (`/webhook/chatwoot`)**
  - The endpoint is designed around the `message_created` webhook payload shape
    described in the Chatwoot docs: [`How to use webhooks?`](https://www.chatwoot.com/hc/user-guide/articles/1677693021-how-to-use-webhooks).
  - Expected minimal body:
    ```json
    {
      "event": "message_created",
      "id": 1,
      "content": "Hi",
      "message_type": "incoming",
      "content_type": "text"
    }
    ```
  - Behaviour:
    - If `event != "message_created"` or `message_type != "incoming"`: the event is ignored and `{ "status": "ignored" }` is returned.
    - If **human mode is OFF**:
      - The backend calls `generate_bot_response` (using OpenAI if configured, otherwise a mock AI).
      - It responds with:
        ```json
        {
          "status": "ok",
          "route": "bot",
          "reply": "Bot reply text here"
        }
        ```
    - If **human mode is ON**:
      - The backend does **not** generate a bot reply and returns:
        ```json
        {
          "status": "ok",
          "route": "human",
          "message": "Human mode is ON. No bot reply generated."
        }
        ```
      - The expectation is that a human agent platform (for example Chatwoot’s own UI)
        will take over the conversation and send the reply to the end user.

- **REST API (`/toggle-human`)**
  - Request:
    ```http
    POST /toggle-human
    Content-Type: application/json

    { "enabled": true }
    ```
  - Response:
    ```json
    { "human_mode": true }
    ```
  - The frontend uses this endpoint to switch **Human Mode** ON/OFF and update the UI indicator.

---

### Frontend UI & Behavior

The frontend (`frontend/index.html`) is a **single, self-contained page** focused on the end-user chat.

- **User Chat panel**
  - Chat transcript area with **auto-scroll**.
  - Input box + **Send** button (and Enter to send).
  - Messages show:
    - **Sender label** (You / Bot / System)
    - **Timestamp** (local time)
    - Different colors for user, bot, and system messages.
  - When you send a message:
    - The frontend immediately shows it as **You**.
    - It then posts a **Chatwoot-style webhook payload** to `POST /webhook/chatwoot`.
    - If the backend responds with `{ route: "bot", reply: "..." }`, the reply is shown as **Bot**.
    - If the backend responds with `{ route: "human" }`, a small system note is added saying the
      message was handed off to a human agent platform.

- **Human toggle**
  - A prominent **“Human”** pill-style toggle in the header.
  - Visual indicator shows:
    - **Human Mode: ON** (green) or **OFF** (red).
  - Clicking the toggle:
    - Calls `POST /toggle-human` with the desired state.
    - Updates the UI and appends a system message describing the change.

> In a real integration, you would configure `POST /webhook/chatwoot` as the webhook URL inside
> Chatwoot (Settings → Integrations → Webhooks) as described in their docs
> [`How to use webhooks?`](https://www.chatwoot.com/hc/user-guide/articles/1677693021-how-to-use-webhooks).

---

### How to Demo the Human Handoff

1. **Start the backend**
   - In `backend/`, with your venv activated:
     ```bash
     uvicorn backend.main:app --reload
     ```

2. **Open the UI**
   - Go to `http://localhost:8000/` in your browser.

3. **Bot mode demo (Human Mode OFF)**
   - Ensure the header indicator shows **“Human Mode: OFF”**.
   - In the **User Chat** panel, send a few messages.
   - You should see **mock AI** responses labeled as **Bot** with timestamps.

4. **Connect a human agent**
   - In the **Agent Console** (right panel), click **“Connect Agent”**.
   - The status should change to **“Agent: Connected”**.

5. **Enable Human Mode**
   - Click the **“Human”** toggle in the header.
   - The indicator should switch to **“Human Mode: ON”** and turn green.

6. **Send a message as a user (routed to human)**
   - In the **User Chat** panel, send a message.
   - The user sees a system note saying it was routed to a human agent.
   - In the **Agent Console**, you should now see a new entry with:
     - The **user’s message text**
     - A generated **user ID** (e.g., `user-1`)

7. **Reply as the human agent**
   - In the agent entry, click **“Use”** to populate the *User ID* field.
   - Type a reply message in the **reply input**.
   - Click **“Send Reply”**.
   - The user should immediately see a new message labeled **“Human Agent”** in their chat, with a timestamp.

8. **Switch back to bot mode**
   - Turn **Human Mode** OFF via the toggle.
   - New user messages will again be handled by the **mock AI bot**.

---

### Notes & Constraints

- **No external UI frameworks** – the frontend is pure HTML/CSS/vanilla JS.
- **No paid APIs required** – by default the bot is a **mock AI** implemented in Python in `generate_bot_response`.
- **In-memory only** – all state (the `human_mode` flag) is stored in memory:
  - Restarting the app clears all state.

---

### Optional: Use OpenAI for Bot Responses

By default, the bot uses a **local mock response** so the demo works without any external services or costs.  
If you want to power the bot with a real LLM via the **OpenAI API**:

1. Install dependencies (already in `backend/requirements.txt`):
   - `openai`
   - `python-dotenv`
2. Create a `.env` file in the `backend/` directory (same place you run `uvicorn`) with:
   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   # Optional: override the default model (defaults to gpt-4o-mini)
   OPENAI_MODEL=gpt-4o-mini
   ```
3. Restart the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

When `OPENAI_API_KEY` is present:

- `generate_bot_response` will use the **OpenAI Chat Completions API** with the model from `OPENAI_MODEL`.
- If the OpenAI request fails for any reason (network, auth, etc.), the code automatically falls back to the **mock AI** response so the POC keeps working.

> Important: Using the OpenAI API may incur costs depending on your account and model choice.  
> The integration is **optional** and entirely disabled unless you set `OPENAI_API_KEY`.

---

### Optional Extensions (Ideas)

These are not implemented, but the architecture makes them straightforward:

- Persist conversations (e.g., to a database).
- Per-user **human mode** instead of a single global flag.
- More advanced **agent routing** (e.g., round-robin assignment, skills-based routing).
- Richer UI (message grouping, avatars, typing indicators, etc.).


