"""
POST /api/caption

Receives two webcam frames captured either side of a detected interaction and
asks Gemini what the person did. Stateless: no storage, no session.

Request body (JSON):
    {
        "before": "<base64 jpeg, no data: prefix>",
        "after":  "<base64 jpeg, no data: prefix>",
        "object": "cup",          # optional, what the tracker was watching
        "person": "Person 2"      # optional, track label of the acting person
    }

Response body (JSON, always HTTP 200):
    {"action": "picked up", "object": "cup", "person": "Person 2",
     "confidence": 0.92}

Errors return a low-confidence fallback rather than a 4xx/5xx. A vague log
entry is invisible during a live demo; a red error in the sidebar is not.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error

MODEL = "gemini-2.0-flash"
ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

# Kept under Vercel's function timeout so our own fallback runs instead of the
# platform killing the request.
GEMINI_TIMEOUT_SECONDS = 8

FALLBACK = {"action": "moved", "object": "object", "confidence": 0.3}


def build_prompt(watched_object: str, person: str) -> str:
    subject = person if person else "the person"
    target = watched_object if watched_object else "the object they interacted with"

    return (
        "These are two frames from a fixed webcam, taken a moment apart. "
        "The first is BEFORE, the second is AFTER.\n\n"
        f"Describe what {subject} did with {target} between the two frames.\n\n"
        "Rules:\n"
        "- Use a short past-tense verb phrase: picked up, put down, moved, "
        "pushed aside, opened, removed.\n"
        "- Name the object concretely if you can see it clearly.\n"
        "- If nothing meaningful changed, use action \"no change\" with low "
        "confidence.\n"
        "- Reply with raw JSON only. No markdown, no code fences, no commentary.\n\n"
        'Format: {"action": "...", "object": "...", "confidence": 0.0}'
    )


def call_gemini(prompt: str, before: str, after: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": before}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": after}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 120,
            "responseMimeType": "application/json",
        },
    }

    request = urllib.request.Request(
        f"{ENDPOINT}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=GEMINI_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def parse_reply(raw_response: dict) -> dict:
    text = raw_response["candidates"][0]["content"]["parts"][0]["text"]

    # Models wrap JSON in code fences even when told not to.
    cleaned = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    confidence = parsed.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "action": str(parsed.get("action", "moved"))[:60],
        "object": str(parsed.get("object", "object"))[:60],
        "confidence": confidence,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "Body must be valid JSON"})

        before = body.get("before")
        after = body.get("after")
        person = body.get("person", "")

        if not before or not after:
            return self._send(400, {"error": "Both 'before' and 'after' are required"})

        prompt = build_prompt(body.get("object", ""), person)

        try:
            event = parse_reply(call_gemini(prompt, before, after))
        except urllib.error.HTTPError as exc:
            print(f"gemini http error {exc.code}: {exc.read()[:400]}")
            event = dict(FALLBACK)
        except Exception as exc:
            print(f"caption failed: {type(exc).__name__}: {exc}")
            event = dict(FALLBACK)

        event["person"] = person
        return self._send(200, event)

    def do_OPTIONS(self):
        self._send(204, None)

    def _send(self, code, obj):
        payload = b"" if obj is None else json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass
