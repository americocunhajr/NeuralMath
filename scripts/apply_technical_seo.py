from pathlib import Path
import html
import json
import re

DATE = "2026-10-07"
INDEXNOW_KEY = "9b83762f5d4f4b36a7e2147c8d6a31e1"

LANGUAGES = [
    ("en", "en", "docs/index.html", "https://neuralmath.org/"),
    ("pt", "pt-BR", "docs/pt/index.html", "https://neuralmath.org/pt/"),
    ("es", "es", "docs/es/index.html", "https://neuralmath.org/es/"),
    ("fr", "fr", "docs/fr/index.html", "https://neuralmath.org/fr/"),
    ("it", "it", "docs/it/index.html", "https://neuralmath.org/it/"),
    ("de", "de", "docs/de/index.html", "https://neuralmath.org/de/"),
    ("ar", "ar", "docs/ar/index.html", "https://neuralmath.org/ar/"),
    ("bn", "bn", "docs/bn/index.html", "https://neuralmath.org/bn/"),
    ("cs", "cs", "docs/cs/index.html", "https://neuralmath.org/cs/"),
    ("el", "el", "docs/el/index.html", "https://neuralmath.org/el/"),
    ("fa", "fa", "docs/fa/index.html", "https://neuralmath.org/fa/"),
    ("hi", "hi", "docs/hi/index.html", "https://neuralmath.org/hi/"),
    ("hu", "hu", "docs/hu/index.html", "https://neuralmath.org/hu/"),
    ("id", "id", "docs/id/index.html", "https://neuralmath.org/id/"),
    ("ja", "ja", "docs/ja/index.html", "https://neuralmath.org/ja/"),
    ("ko", "ko", "docs/ko/index.html", "https://neuralmath.org/ko/"),
    ("nl", "nl", "docs/nl/index.html", "https://neuralmath.org/nl/"),
    ("pl", "pl", "docs/pl/index.html", "https://neuralmath.org/pl/"),
    ("ro", "ro", "docs/ro/index.html", "https://neuralmath.org/ro/"),
    ("ru", "ru", "docs/ru/index.html", "https://neuralmath.org/ru/"),
    ("sv", "sv", "docs/sv/index.html", "https://neuralmath.org/sv/"),
    ("tr", "tr", "docs/tr/index.html", "https://neuralmath.org/tr/"),
    ("uk", "uk", "docs/uk/index.html", "https://neuralmath.org/uk/"),
    ("ur", "ur", "docs/ur/index.html", "https://neuralmath.org/ur/"),
    ("vi", "vi", "docs/vi/index.html", "https://neuralmath.org/vi/"),
    ("zh", "zh-Hans", "docs/zh/index.html", "https://neuralmath.org/zh/"),
]

PERSON_ID = "https://neuralmath.org/#americo-cunha-jr"
WEBSITE_ID = "https://neuralmath.org/#website"

def esc(value):
    return html.escape(value, quote=True)

for code, bcp47, filename, canonical in LANGUAGES:
    path = Path(filename)
    source = path.read_text(encoding="utf-8")
    body = source[source.index("<body>"):]
    source = re.sub(r"<!-- SEO-CONFIG-START -->.*?<!-- SEO-CONFIG-END -->\s*", "", source, flags=re.S)
    source = re.sub(r"<!-- SEO-META-START -->.*?<!-- SEO-META-END -->\s*", "", source, flags=re.S)

    title = re.search(r"<title>(.*?)</title>", source, flags=re.S).group(1)
    headline = title.split("—", 1)[1].strip() if "—" in title else title.strip()
    description = re.search(r'<meta content="([^"]*)" name="description"/>', source, flags=re.S).group(1)
    pdf_url = re.search(r'<meta name="citation_pdf_url" content="([^"]*)"/>', source).group(1)
    image = f"https://neuralmath.org/assets/img/{code}/Fig07.png"

    person = {
        "@type": "Person",
        "@id": PERSON_ID,
        "name": "Americo Cunha Jr.",
        "sameAs": [
            "https://orcid.org/0000-0002-8342-0363",
            "https://github.com/americocunhajr",
        ],
    }
    article = {
        "@type": "Article",
        "@id": canonical + "#article",
        "headline": headline,
        "description": description,
        "url": canonical,
        "mainEntityOfPage": canonical,
        "inLanguage": bcp47,
        "image": image,
        "datePublished": DATE,
        "dateModified": DATE,
        "author": {"@id": PERSON_ID},
        "publisher": {"@id": PERSON_ID},
        "isAccessibleForFree": True,
        "encoding": {
            "@type": "MediaObject",
            "contentUrl": pdf_url,
            "encodingFormat": "application/pdf",
        },
        "isPartOf": {"@id": WEBSITE_ID},
    }
    graph = [person, article]
    if code == "en":
        graph.insert(0, {
            "@type": "WebSite",
            "@id": WEBSITE_ID,
            "url": "https://neuralmath.org/",
            "name": "NeuralMath",
            "author": {"@id": PERSON_ID},
        })
    structured = {"@context": "https://schema.org", "@graph": graph}

    block = f"""<!-- SEO-META-START -->\n<!-- SEO-CONFIG-START -->
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
<script type="application/ld+json">{json.dumps(structured, ensure_ascii=False, separators=(",", ":"))}</script>
<!-- SEO-CONFIG-END -->
"""
    insert_at = source.index("<link ")
    source = source[:insert_at] + block + source[insert_at:]
    assert source[source.index("<body>"):] == body, f"body changed: {filename}"
    path.write_text(source, encoding="utf-8")

sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
]
for _, _, _, canonical in LANGUAGES:
    sitemap.append(f"  <url><loc>{canonical}</loc><lastmod>{DATE}</lastmod></url>")
sitemap.append("</urlset>")
Path("docs/sitemap.xml").write_text("\n".join(sitemap) + "\n", encoding="utf-8")
Path(f"docs/{INDEXNOW_KEY}.txt").write_text(INDEXNOW_KEY + "\n", encoding="utf-8")
print("Applied consistent technical SEO metadata and JSON-LD to all 26 pages.")
