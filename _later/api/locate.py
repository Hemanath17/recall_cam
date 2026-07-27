"""
POST /api/locate

Open-vocabulary object detection. Send one frame and a plain-English object
name; Gemini returns bounding boxes for every instance it finds. Unlike the
COCO detector in the browser, this is not limited to a fixed class list, so
"pen", "stapler", "blue notebook" all work.

Slow (about a second) and metered, so it is called on an interval rather than
per frame. The browser tracks whatever it finds in between.

Request body (JSON):
    {
        "image": "<base64 jpeg, no data: prefix>",
        "target": "pen"          # what to look for
    }

Response body (JSON, always HTTP 200):
    {
        "objects": [
            {"label": "pen", "x": 0.31, "y": 0.55, "w": 0.12, "h": 0.06,
             "confidence": 0.9}
        ]
    }

Coordinates are fractions of image width and height, origin top-left. The
caller multiplies by canvas size. Gemini natively returns
[ymin, xmin, ymax, xmax] scaled 0-1000; that conversion happens here so the
frontend never has to know about it.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error

MODEL = "gemini-2.5-flash"
ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

GEMINI_TIMEOUT_SECONDS = 8

# Gemini emits box coordinates on this scale.
GEMINI_COORD_SCALE = 1000.0


def build_prompt(target: str) -> str:
    return (
        f"Find every {target} visible in this image.\n\n"
        "Return a JSON array. One entry per instance. Each entry:\n"
        '  {"label": "<short name>", "box_2d": [ymin, xmin, ymax, xmax], '
        '"confidence": <0.0 to 1.0>}\n\n'
        "Rules:\n"
        "- box_2d uses integers from 0 to 1000, in the order "
        "ymin, xmin, ymax, xmax.\n"
        "- label is two or three words describing that specific instance, "
        "for example \"blue ballpoint pen\".\n"
        "- Return an empty array if nothing matches. Do not invent objects.\n"
        "- Reply with raw JSON only. No markdown, no code fences, no prose."
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
            "temperature": 0.0,
            "maxOutputTokens": 800,
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_reply(raw_response: dict, target: str) -> list:
    text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    # Tolerate the model wrapping the array in an object.
    if isinstance(parsed, dict):
        for key in ("objects", "results", "detections", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = []

    if not isinstance(parsed, list):
        return []

    objects = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue

        box = entry.get("box_2d") or entry.get("box") or entry.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            ymin, xmin, ymax, xmax = (float(v) / GEMINI_COORD_SCALE for v in box)
        except (TypeError, ValueError):
            continue

        # Guard against the model swapping min and max.
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        x = clamp01(xmin)
        y = clamp01(ymin)
        w = clamp01(xmax) - x
        h = clamp01(ymax) - y

        # Degenerate or full-frame boxes are almost always hallucinations.
        if w <= 0.01 or h <= 0.01 or (w > 0.97 and h > 0.97):
            continue

        try:
            confidence = clamp01(float(entry.get("confidence", 0.8)))
        except (TypeError, ValueError):
            confidence = 0.8

        objects.append(
            {
                "label": str(entry.get("label", target))[:60].lower(),
                "x": round(x, 4),
                "y": round(y, 4),
                "w": round(w, 4),
                "h": round(h, 4),
                "confidence": confidence,
            }
        )

    return objects


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or "{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "Body must be valid JSON"})

        image = body.get("image")
        target = str(body.get("target", "")).strip()

        if not image:
            return self._send(400, {"error": "'image' is required"})
        if not target:
            return self._send(400, {"error": "'target' is required"})

        try:
            objects = parse_reply(call_gemini(build_prompt(target), image), target)
        except urllib.error.HTTPError as exc:
            print(f"gemini http error {exc.code}: {exc.read()[:400]}")
            objects = []
        except Exception as exc:
            print(f"locate failed: {type(exc).__name__}: {exc}")
            objects = []

        return self._send(200, {"objects": objects})

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