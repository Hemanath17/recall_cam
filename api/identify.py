"""
POST /api/identify

Receives a single cropped image of one tracked object and returns a specific,
human-readable name for it. Called once per object track, then never again for
that track, so it stays cheap.

Crop the image in the browser before sending. If you send the full frame,
Gemini will describe the most visually prominent thing in the scene rather than
the object you are tracking.

Request body (JSON):
    {
        "image": "<base64 jpeg, no data: prefix>",
        "label": "cup"        # optional COCO class as a hint
    }

Response body (JSON, always HTTP 200):
    {"name": "red ceramic mug"}

On failure it echoes the COCO label back, so the caller always has a usable
name and the UI never shows an error state.
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

GEMINI_TIMEOUT_SECONDS = 8


def build_prompt(label: str) -> str:
    hint = (
        f"An object detector classified it as \"{label}\". "
        if label
        else ""
    )

    return (
        "This is a close crop of a single object from a webcam. "
        f"{hint}"
        "Name this specific object the way its owner would describe it.\n\n"
        "Rules:\n"
        "- Two to four words, lowercase, no article.\n"
        "- Include the distinguishing colour or material if it is visible, "
        "for example \"red ceramic mug\" or \"black water bottle\".\n"
        "- Describe only the object, not the background or the person.\n"
        "- If the crop is too blurry or ambiguous to tell, return the detector "
        "label unchanged.\n"
        "- Reply with raw JSON only. No markdown, no code fences.\n\n"
        'Format: {"name": "..."}'
    )


def call_gemini(prompt: str, image: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 40,
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


def parse_reply(raw_response: dict, fallback: str) -> str:
    text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    name = str(parsed.get("name", "")).strip().lower()
    if not name:
        return fallback

    # Guard against the model returning a sentence instead of a label.
    words = name.split()
    if len(words) > 5:
        name = " ".join(words[:5])

    return name[:60]


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or "{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "Body must be valid JSON"})

        image = body.get("image")
        label = str(body.get("label", "")).strip()

        if not image:
            return self._send(400, {"error": "'image' is required"})

        fallback = label if label else "object"

        try:
            name = parse_reply(call_gemini(build_prompt(label), image), fallback)
        except urllib.error.HTTPError as exc:
            print(f"gemini http error {exc.code}: {exc.read()[:400]}")
            name = fallback
        except Exception as exc:
            print(f"identify failed: {type(exc).__name__}: {exc}")
            name = fallback

        return self._send(200, {"name": name})

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