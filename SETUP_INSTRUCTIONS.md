# 🚀 Complete Setup Instructions

## ✅ CRITICAL: The Real Chatwoot Integration is NOW LIVE

Your backend **NOW actually talks to Chatwoot**. Here's what changed:

### What Was Wrong Before
- ❌ No API calls to Chatwoot
- ❌ Just simulating handoff in UI
- ❌ Messages never appeared in Chatwoot inbox

### What Works Now
- ✅ Real contact/conversation creation
- ✅ Real message sending to Chatwoot inbox
- ✅ Real webhook receiver for agent replies
- ✅ Comprehensive logging for debugging

---

## 📋 Step-by-Step Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Create `.env` File in `backend/` Directory

Create a file named `.env` inside the `backend/` folder with this content:

```bash
# OpenAI Configuration (optional - for bot mode)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini

# Chatwoot Configuration (REQUIRED for human handoff)
CHATWOOT_BASE_URL=https://app.chatwoot.com
CHATWOOT_API_TOKEN=YOUR_API_ACCESS_TOKEN
CHATWOOT_ACCOUNT_ID=143978
CHATWOOT_INBOX_ID=YOUR_INBOX_ID
```

### 3. Get Your Chatwoot Credentials

#### A) Get API Token
1. Go to Chatwoot → Profile Settings (bottom left, your name)
2. Click "Access Token" or "Profile Settings"
3. Copy your **API Access Token**
4. Paste it in `.env` as `CHATWOOT_API_TOKEN`

#### B) Get Inbox ID
1. Go to Settings → Inboxes
2. Click on your **API inbox** (or create one if you don't have it)
3. Look at the URL: `app.chatwoot.com/app/accounts/143978/settings/inboxes/{INBOX_ID}`
4. Copy the `INBOX_ID` number
5. Paste it in `.env` as `CHATWOOT_INBOX_ID`

**Important:** You MUST use an **API channel inbox**, not Website/Email/WhatsApp.

#### How to Create an API Inbox (if needed)
1. Settings → Inboxes → Add Inbox
2. Select "API" as the channel
3. Give it a name (e.g., "Chatbot API")
4. Set a callback URL (can be anything for now, like `https://example.com`)
5. Click "Create API Channel"
6. Note the inbox ID from the URL

### 4. Configure Chatwoot Webhook (Important!)

For agent replies to reach your UI, you need to tell Chatwoot where to send them:

1. Go to Chatwoot: **Settings → Integrations → Webhooks**
2. Click **"Add new webhook"**
3. **URL:** `https://your-ngrok-or-domain.com/webhook/chatwoot`
   - If testing locally, use ngrok (see below)
4. **Events:** Check ✅ **message_created**
5. Click **"Add"**

### 5. Run the Backend

```bash
cd backend
uvicorn backend.main:app --reload --log-level info
```

Watch the logs carefully! You'll see:
```
=== Chatwoot Configuration ===
BASE_URL: https://app.chatwoot.com
ACCOUNT_ID: 143978
INBOX_ID: 123
API_TOKEN: ***
==============================
```

### 6. Expose to Internet (for Webhook Testing)

If testing locally, use **ngrok**:

```bash
ngrok http 8000
```

Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`) and use it in Step 4 above.

---

## 🔍 How to Debug API Failures

The new code has **extensive logging**. Watch your terminal for:

### ✅ Success Messages
```
✅ Contact created/found: contact_id=456
✅ Conversation created: conversation_id=789
✅ Message sent to Chatwoot: message_id=1011
👤 Agent reply detected
✅ Queueing agent reply for user=web-user
```

### ❌ Error Messages
```
❌ Failed to create contact for user=web-user: 401 Unauthorized
```

Common errors:

| Error | Meaning | Fix |
|-------|---------|-----|
| **401 Unauthorized** | Wrong API token | Check `CHATWOOT_API_TOKEN` |
| **404 Not Found** | Wrong account/inbox ID | Verify `CHATWOOT_ACCOUNT_ID` and `CHATWOOT_INBOX_ID` |
| **422 Unprocessable** | Wrong inbox type | Must use API channel inbox |
| **Connection refused** | Wrong base URL | Check `CHATWOOT_BASE_URL` |

---

## 🧪 Test the Complete Flow

### 1. Bot Mode (No Chatwoot)
1. Open `http://localhost:8000`
2. Ensure "HUMAN MODE: OFF" (default)
3. Send a message
4. You should get a bot reply immediately

### 2. Human Mode (Chatwoot Integration)
1. Click the **HUMAN** toggle (turns blue, says "HUMAN MODE: ON")
2. Watch backend logs - you should see:
   ```
   Creating new Chatwoot conversation for user=web-user
   POST https://app.chatwoot.com/api/v1/accounts/143978/contacts
   ✅ Contact created/found: contact_id=...
   POST https://app.chatwoot.com/api/v1/accounts/143978/conversations
   ✅ Conversation created: conversation_id=...
   ```
3. Send a message (e.g., "Hello from POC")
4. Backend logs should show:
   ```
   🙋 HUMAN MODE: Forwarding message to Chatwoot
   POST https://app.chatwoot.com/.../conversations/.../messages
   ✅ Message sent to Chatwoot: message_id=...
   ```
5. **Go to your Chatwoot inbox** - you should now see the conversation!
6. Reply as an agent in Chatwoot
7. Backend should receive webhook:
   ```
   📨 Webhook from ...: event=message_created
   👤 Agent reply detected
   ✅ Queueing agent reply for user=web-user
   ```
8. Your web UI should show the agent reply within 2 seconds (polling interval)

---

## 🐛 Still Not Working?

### Test API Credentials Manually

Run this in terminal (replace values):

```bash
curl -X POST https://app.chatwoot.com/api/v1/accounts/143978/contacts \
  -H "api_access_token: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"test-user","identifier":"test-user"}'
```

**If this fails**, your credentials are wrong.

**If this succeeds**, check:
1. Is your inbox an API channel inbox?
2. Are you using the correct inbox ID?
3. Is your webhook URL reachable from Chatwoot?

---

## 📝 Architecture Summary

```
┌─────────────┐
│   Browser   │
│   (User)    │
└─────┬───────┘
      │ POST /message
      ▼
┌─────────────┐
│   FastAPI   │ (Human mode: OFF)
│   Backend   │ ──────────────────► Generate bot reply
└─────┬───────┘                      (OpenAI or mock)
      │
      │ (Human mode: ON)
      ├─────────► POST /contacts (Chatwoot)
      ├─────────► POST /conversations (Chatwoot)
      └─────────► POST /conversations/{id}/messages (Chatwoot)
                                    │
                                    ▼
                          ┌─────────────────┐
                          │ Chatwoot Inbox  │
                          │ (Agent sees it) │
                          └─────────┬───────┘
                                    │ Agent replies
                                    ▼
                          POST /webhook/chatwoot (your backend)
                                    │
                                    ▼
                          Browser polls /poll-human-messages
                                    │
                                    ▼
                          Agent reply appears in UI
```

---

## ✅ Checklist

Before reporting issues, confirm:

- [ ] `.env` file exists in `backend/` folder
- [ ] All 4 Chatwoot vars are set correctly
- [ ] API token is valid (test with curl)
- [ ] Account ID matches your Chatwoot URL
- [ ] Inbox is an **API channel** (not Website/Email/etc)
- [ ] Inbox ID is correct
- [ ] Webhook is configured in Chatwoot
- [ ] Webhook URL is publicly accessible (use ngrok if local)
- [ ] Backend logs show config on startup

---

## 🎯 What You Should See

### Backend Logs (on startup)
```
=== Chatwoot Configuration ===
BASE_URL: https://app.chatwoot.com
ACCOUNT_ID: 143978
INBOX_ID: 456
API_TOKEN: ***
==============================
```

### Backend Logs (when sending message in human mode)
```
Human mode toggle requested from 127.0.0.1: enabled=True
Human mode is now True
Creating new Chatwoot conversation for user=web-user
POST https://app.chatwoot.com/api/v1/accounts/143978/contacts
✅ Contact created/found: contact_id=789
POST https://app.chatwoot.com/api/v1/accounts/143978/conversations
✅ Conversation created: conversation_id=1234
Message from 127.0.0.1 (user=web-user): hello from poc
🙋 HUMAN MODE: Forwarding message to Chatwoot
POST https://app.chatwoot.com/.../conversations/1234/messages
✅ Message sent to Chatwoot: message_id=5678
```

### Chatwoot Inbox
- New conversation appears
- Contact name: "web-user"
- Message: "hello from poc"
- You can reply as agent

### Frontend (after agent replies)
- Agent message appears with green background
- Label: "Human Agent"
- Content: whatever the agent typed

---

## 🚀 You're Done!

The system now **fully integrates** with Chatwoot. Messages go from your UI → Chatwoot inbox, and agent replies come back to your UI in real time.

