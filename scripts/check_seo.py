from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

pages = [Path("docs/index.html")] + sorted(Path("docs").glob("??/index.html"))
assert len(pages) == 26

for page in pages:
    source = page.read_text(encoding="utf-8")
    assert 'name="robots" content="index, follow' in source, page
    for tag in ["og:type", "og:site_name", "og:title", "og:description", "og:url", "og:image"]:
        assert f'property="{tag}"' in source, (page, tag)
    for tag in ["twitter:card", "twitter:title", "twitter:description", "twitter:image"]:
        assert f'name="{tag}"' in source, (page, tag)
    assert source.count('rel="alternate"') == 27, page
    assert 'rel="canonical"' in source, page
    assert "noindex" not in source.lower(), page

    block = re.search(r'<script type="application/ld\+json">(.*?)</script>', source, re.S)
    assert block, page
    data = json.loads(block.group(1))
    assert data.get("@context") == "https://schema.org", page

    pdf = re.search(r'<meta name="citation_pdf_url" content="([^"]+)"/>', source)
    assert pdf and pdf.group(1).startswith("https://neuralmath.org/pdf/NeuralMath_"), page
    local_pdf = Path("docs/pdf") / pdf.group(1).rsplit("/", 1)[-1]
    assert local_pdf.exists() and local_pdf.stat().st_size > 100000, local_pdf

root = ET.parse("docs/sitemap.xml").getroot()
ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
urls = root.findall("s:url", ns)
assert len(urls) == 26
for entry in urls:
    assert entry.find("s:loc", ns) is not None
    lastmod = entry.find("s:lastmod", ns)
    assert lastmod is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", lastmod.text or "")

robots = Path("docs/robots.txt").read_text(encoding="utf-8")
assert "Sitemap: https://neuralmath.org/sitemap.xml" in robots

print("SEO checks passed for all 26 language editions.")
