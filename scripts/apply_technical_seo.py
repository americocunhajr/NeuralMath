from pathlib import Path
import re
import html

DATE = "2026-10-07"
INDEXNOW_KEY = "9b83762f5d4f4b36a7e2147c8d6a31e1"

LANGUAGES = [
    ("en", "docs/index.html", "https://neuralmath.org/"),
    ("pt", "docs/pt/index.html", "https://neuralmath.org/pt/"),
    ("es", "docs/es/index.html", "https://neuralmath.org/es/"),
    ("fr", "docs/fr/index.html", "https://neuralmath.org/fr/"),
    ("it", "docs/it/index.html", "https://neuralmath.org/it/"),
    ("de", "docs/de/index.html", "https://neuralmath.org/de/"),
    ("ar", "docs/ar/index.html", "https://neuralmath.org/ar/"),
    ("bn", "docs/bn/index.html", "https://neuralmath.org/bn/"),
    ("cs", "docs/cs/index.html", "https://neuralmath.org/cs/"),
    ("el", "docs/el/index.html", "https://neuralmath.org/el/"),
    ("fa", "docs/fa/index.html", "https://neuralmath.org/fa/"),
    ("hi", "docs/hi/index.html", "https://neuralmath.org/hi/"),
    ("hu", "docs/hu/index.html", "https://neuralmath.org/hu/"),
    ("id", "docs/id/index.html", "https://neuralmath.org/id/"),
    ("ja", "docs/ja/index.html", "https://neuralmath.org/ja/"),
    ("ko", "docs/ko/index.html", "https://neuralmath.org/ko/"),
    ("nl", "docs/nl/index.html", "https://neuralmath.org/nl/"),
    ("pl", "docs/pl/index.html", "https://neuralmath.org/pl/"),
    ("ro", "docs/ro/index.html", "https://neuralmath.org/ro/"),
    ("ru", "docs/ru/index.html", "https://neuralmath.org/ru/"),
    ("sv", "docs/sv/index.html", "https://neuralmath.org/sv/"),
    ("tr", "docs/tr/index.html", "https://neuralmath.org/tr/"),
    ("uk", "docs/uk/index.html", "https://neuralmath.org/uk/"),
    ("ur", "docs/ur/index.html", "https://neuralmath.org/ur/"),
    ("vi", "docs/vi/index.html", "https://neuralmath.org/vi/"),
    ("zh", "docs/zh/index.html", "https://neuralmath.org/zh/"),
]

def esc(value):
    return html.escape(value, quote=True)

for code, filename, canonical in LANGUAGES:
    path = Path(filename)
    source = path.read_text(encoding="utf-8")
    body = source[source.index("<body>"):]
    source = re.sub(
        r"<!-- SEO-META-START -->.*?<!-- SEO-META-END -->\s*",
        "",
        source,
        flags=re.S,
    )
    title = re.search(r"<title>(.*?)</title>", source, flags=re.S).group(1)
    headline = title.split("—", 1)[1].strip() if "—" in title else title.strip()
    description = re.search(
        r'<meta content="([^"]*)" name="description"/>', source, flags=re.S
    ).group(1)
    image = f"https://neuralmath.org/assets/img/{code}/Fig07.png"
    block = f"""<!-- SEO-META-START -->
<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1"/>
<meta property="og:type" content="article"/>
<meta property="og:site_name" content="NeuralMath"/>
<meta property="og:title" content="{esc(headline)}"/>
<meta property="og:description" content="{esc(description)}"/>
<meta property="og:url" content="{canonical}"/>
<meta property="og:image" content="{image}"/>
<meta property="og:image:alt" content="{esc(headline)}"/>
<meta property="article:published_time" content="{DATE}"/>
<meta property="article:modified_time" content="{DATE}"/>
<meta name="twitter:card" content="summary_large_image"/>
<meta name="twitter:title" content="{esc(headline)}"/>
<meta name="twitter:description" content="{esc(description)}"/>
<meta name="twitter:image" content="{image}"/>
<!-- SEO-META-END -->
"""
    insert_at = source.index("<link ")
    source = source[:insert_at] + block + source[insert_at:]
    assert source[source.index("<body>"):] == body, f"body changed: {filename}"
    path.write_text(source, encoding="utf-8")

sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
]
for _, _, canonical in LANGUAGES:
    sitemap.append(f"  <url><loc>{canonical}</loc><lastmod>{DATE}</lastmod></url>")
sitemap.append("</urlset>")
Path("docs/sitemap.xml").write_text("\n".join(sitemap) + "\n", encoding="utf-8")
Path(f"docs/{INDEXNOW_KEY}.txt").write_text(INDEXNOW_KEY + "\n", encoding="utf-8")
print("Applied technical SEO metadata to 26 pages without changing page bodies.")
