from pathlib import Path
import json
import time
import urllib.request
import xml.etree.ElementTree as ET

KEY = "9b83762f5d4f4b36a7e2147c8d6a31e1"
KEY_URL = f"https://neuralmath.org/{KEY}.txt"

for _ in range(24):
    try:
        with urllib.request.urlopen(KEY_URL, timeout=15) as response:
            if response.read().decode("utf-8").strip() == KEY:
                break
    except Exception:
        pass
    time.sleep(15)
else:
    raise SystemExit("IndexNow key file was not available after deployment wait.")

root = ET.parse("docs/sitemap.xml").getroot()
ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
urls = [node.text for node in root.findall("s:url/s:loc", ns)]
assert len(urls) == 26

payload = json.dumps({
    "host": "neuralmath.org",
    "key": KEY,
    "keyLocation": KEY_URL,
    "urlList": urls,
}).encode("utf-8")

request = urllib.request.Request(
    "https://api.indexnow.org/indexnow",
    data=payload,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=30) as response:
    status = response.status
    body = response.read().decode("utf-8", errors="replace")

if status not in (200, 202):
    raise SystemExit(f"IndexNow submission failed with HTTP {status}: {body}")

print(f"IndexNow accepted {len(urls)} URLs with HTTP {status}.")
