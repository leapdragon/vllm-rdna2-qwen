#!/usr/bin/env python3
"""Vision smoke test for the Flash-Next fork (VISION=1): synthetic images with known content
sent through the OpenAI chat API as base64 data URLs. No downloads, deterministic answers.

  1. colour + shape        (a red circle on white)             -> "red", "circle"
  2. rendered text / OCR   ("SUNRISE 42" in large black type)  -> "SUNRISE", "42"
  3. counting              (5 blue squares)                     -> "5" / "five"
  4. quadrant colours      (4 coloured quadrants, ask top-left) -> "green"
  5. two images in one prompt (which has the triangle?)         -> "second"/"2"
  6. large image (1800x1400, a big yellow star)                 -> "yellow", "star"
  7. over the per-prompt limit (5 images)                      -> HTTP 400, not a crash
  8. text-only request afterwards                              -> still correct
Each check prints the model's answer, first-token latency and the completion size.
Usage: vision_test.py [--base URL] [--limit N] [--save-dir DIR]
"""
import argparse, base64, io, json, sys, time, urllib.error, urllib.request
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8000")
ap.add_argument("--limit", type=int, default=4, help="configured images per prompt")
ap.add_argument("--save-dir", default=None, help="also write the generated PNGs here")
ap.add_argument("--strict-limit", action="store_true",
                help="expect over-limit prompts to be rejected with 400 (MM_ELIDE=0 servers)")
args = ap.parse_args()


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def img_red_circle():
    im = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(im).ellipse((128, 128, 384, 384), fill=(220, 20, 20))
    return im


def img_text():
    im = Image.new("RGB", (768, 256), "white")
    ImageDraw.Draw(im).text((40, 60), "SUNRISE 42", fill="black", font=font(110))
    return im


def img_count():
    im = Image.new("RGB", (640, 320), "white")
    d = ImageDraw.Draw(im)
    for i in range(5):
        x = 40 + i * 120
        d.rectangle((x, 110, x + 80, 190), fill=(30, 60, 220))
    return im


def img_quadrants():
    im = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 256, 256), fill=(20, 160, 40))       # top-left green
    d.rectangle((256, 0, 512, 256), fill=(220, 20, 20))     # top-right red
    d.rectangle((0, 256, 256, 512), fill=(30, 60, 220))     # bottom-left blue
    d.rectangle((256, 256, 512, 512), fill=(240, 200, 20))  # bottom-right yellow
    return im


def img_triangle():
    im = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(im).polygon([(256, 80), (80, 430), (432, 430)], fill=(120, 30, 160))
    return im


def img_big_star():
    im = Image.new("RGB", (1800, 1400), (245, 245, 255))
    import math
    cx, cy, R, r = 900, 700, 520, 220
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    ImageDraw.Draw(im).polygon(pts, fill=(250, 210, 30))
    return im


def data_url(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(images, question, max_tokens=200):
    content = [{"type": "image_url", "image_url": {"url": data_url(im)}} for im in images]
    content.append({"type": "text", "text": question})
    body = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True}
    req = urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = ""
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            delta = d["choices"][0].get("delta", {}).get("content") or ""
            if delta and ttft is None:
                ttft = time.time() - t0
            text += delta
            n_chunks += 1
    return text.strip(), (ttft or 0.0), time.time() - t0, n_chunks


ok = True
def check(name, answer, needles, ttft, dt, n):
    global ok
    low = answer.lower()
    hit = any(all(w.lower() in low for w in group) if isinstance(group, tuple) else group.lower() in low
              for group in needles)
    print(f"{'PASS' if hit else 'FAIL'} {name}: ttft {ttft:.2f}s, {n} tokens in {dt:.1f}s -> {answer[:110]!r}")
    ok = ok and hit


imgs = {"circle": img_red_circle(), "text": img_text(), "count": img_count(),
        "quad": img_quadrants(), "triangle": img_triangle(), "star": img_big_star()}
if args.save_dir:
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    for k, im in imgs.items():
        im.save(os.path.join(args.save_dir, f"{k}.png"))

a = chat([imgs["circle"]], "What shape is in this image and what colour is it? Answer in a few words.")
check("1 colour+shape", a[0], ["red"], *a[1:])
check("1b shape", a[0], ["circle", "round", "disc"], *a[1:])
a = chat([imgs["text"]], "Transcribe the text in this image exactly.")
check("2 OCR", a[0], [("sunrise", "42")], *a[1:])
a = chat([imgs["count"]], "How many blue squares are in this image? Answer with the number only.")
check("3 counting", a[0], ["5", "five"], *a[1:])
a = chat([imgs["quad"]], "This image is divided into four coloured quadrants. What colour is the top-left quadrant? One word.")
check("4 quadrant", a[0], ["green"], *a[1:])
a = chat([imgs["circle"], imgs["triangle"]], "Two images are attached. Which one contains a triangle, the first or the second? Answer 'first' or 'second'.")
check("5 two images", a[0], ["second", "2nd", "image 2"], *a[1:])
a = chat([imgs["star"]], "What is the large shape in this image and what colour is it? A few words.")
check("6 large image", a[0], [("yellow", "star"), ("gold", "star")], *a[1:])

# 7: over the configured limit. Default server behaviour (MM_ELIDE=1): the request succeeds
# and the OLDEST images are elided -- prove it by putting the only text-bearing image first:
# with 6 images and limit 4, the "SUNRISE 42" image must no longer be visible. --strict-limit
# instead expects the stock HTTP 400.
if args.strict_limit:
    try:
        chat([imgs["circle"]] * (args.limit + 1), "Describe these images.", max_tokens=20)
        print(f"FAIL 7 over-limit ({args.limit + 1} images) was accepted")
        ok = False
    except urllib.error.HTTPError as e:
        print(f"PASS 7 over-limit rejected: HTTP {e.code} {e.read().decode()[:100]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL 7 over-limit raised {type(e).__name__}: {e}")
        ok = False
else:
    six = [imgs["text"], imgs["circle"], imgs["triangle"], imgs["quad"], imgs["count"], imgs["star"]]
    q = "Is there any readable text (words or numbers) in any of these images? Answer yes or no."
    try:
        a = chat(six, q, max_tokens=20)
        check("7 over-limit elides the oldest (no text image left)", a[0], ["no"], *a[1:])
        b = chat([imgs["text"], imgs["circle"], imgs["triangle"], imgs["quad"]], q, max_tokens=20)
        check("7b control at the limit still sees the text image", b[0], ["yes"], *b[1:])
    except urllib.error.HTTPError as e:
        print(f"FAIL 7 over-limit got HTTP {e.code} (elision off?): {e.read().decode()[:100]!r}")
        ok = False

# 8: text still fine
body = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
        "max_tokens": 20, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
with urllib.request.urlopen(urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"}), timeout=300) as r:
    t = json.loads(r.read())["choices"][0]["message"]["content"]
print(f"{'PASS' if '391' in t else 'FAIL'} 8 text-only after images -> {t.strip()!r}")
ok = ok and "391" in t
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
