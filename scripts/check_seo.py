"""Validate the published technical SEO contract without changing content."""

from html.parser import HTMLParser
from pathlib import Path
import json
import xml.etree.ElementTree as ET


ROOT = Path("docs")
DOMAIN = "https://neuralmath.org"
DATE = "2026-10-07"
CODES = "ar bn cs de el en es fa fr hi hu id it ja ko nl pl pt ro ru sv tr uk ur vi zh".split()
KEY = "9b83762f5d4f4b36a7e2147c8d6a31e1"
LANG = {code: code for code in CODES}
LANG.update(pt="pt-BR", zh="zh-Hans")
URL = {code: DOMAIN + ("/" if code == "en" else f"/{code}/") for code in CODES}


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.head = False
        self.body = False
        self.html_lang = None
        self.meta = {}
        self.links = []
        self.anchors = []
        self.jsonld = []
        self.in_jsonld = False
        self.jsonld_text = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html":
            self.html_lang = attrs.get("lang")
        elif tag == "head":
            self.head = True
        elif tag == "body":
            self.body = True
        elif self.head and tag == "meta":
            key = attrs.get("name") or attrs.get("property")
            self.meta.setdefault(key, []).append(attrs.get("content"))
        elif self.head and tag == "link":
            self.links.append(attrs)
        elif self.head and tag == "script" and attrs.get("type") == "application/ld+json":
            self.in_jsonld = True
            self.jsonld_text = ""
        elif self.body and tag == "a":
            self.anchors.append(attrs)

    def handle_data(self, data):
        if self.in_jsonld:
            self.jsonld_text += data

    def handle_endtag(self, tag):
        if tag == "script" and self.in_jsonld:
            self.jsonld.append(json.loads(self.jsonld_text))
            self.in_jsonld = False
        elif tag == "head":
            self.head = False
        elif tag == "body":
            self.body = False

    def one(self, key):
        values = self.meta.get(key, [])
        assert len(values) == 1 and values[0], (key, values)
        return values[0]


def check_page(code):
    path = ROOT / ("index.html" if code == "en" else f"{code}/index.html")
    source = path.read_text(encoding="utf-8")
    assert "noindex" not in source[:source.index("</head>")].lower(), path
    page = Page()
    page.feed(source)
    assert page.html_lang == LANG[code], path
    assert page.one("robots") == "index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1", path

    canonical = [link["href"] for link in page.links if link.get("rel") == "canonical"]
    assert canonical == [URL[code]], (path, canonical)
    alternates = [(link.get("hreflang"), link.get("href")) for link in page.links if link.get("rel") == "alternate"]
    expected = {(LANG[other], URL[other]) for other in CODES} | {("x-default", URL["en"])}
    assert len(alternates) == 27 and set(alternates) == expected, path

    title = page.one("citation_title")
    description = page.one("description")
    image = f"{DOMAIN}/assets/img/{code}/Fig07.png"
    pdf = f"{DOMAIN}/pdf/NeuralMath_{code.upper()}.pdf"
    assert page.one("citation_pdf_url") == pdf, path
    assert (ROOT / "pdf" / f"NeuralMath_{code.upper()}.pdf").stat().st_size > 100000, path
    assert (ROOT / "assets" / "img" / code / "Fig07.png").is_file(), path
    for key, value in {
        "og:type": "article", "og:site_name": "NeuralMath", "og:title": title,
        "og:description": description, "og:url": URL[code], "og:image": image,
        "og:image:alt": title, "article:published_time": DATE,
        "article:modified_time": DATE, "twitter:card": "summary_large_image",
        "twitter:title": title, "twitter:description": description, "twitter:image": image,
    }.items():
        assert page.one(key) == value, (path, key)

    assert len(page.jsonld) == 1, path
    data = page.jsonld[0]
    assert data.get("@context") == "https://schema.org", path
    graph = {item["@type"]: item for item in data["@graph"]}
    assert set(graph) == ({"Person", "Article", "WebSite"} if code == "en" else {"Person", "Article"}), path
    person = graph["Person"]
    person_id = f"{DOMAIN}/#americo-cunha-jr"
    assert person["@id"] == person_id and person["name"] == "Americo Cunha Jr.", path
    assert "https://orcid.org/0000-0002-8342-0363" in person["sameAs"], path
    assert "https://github.com/americocunhajr" in person["sameAs"], path
    article = graph["Article"]
    for key, value in {
        "@id": URL[code] + "#article", "headline": title,
        "description": description, "url": URL[code],
        "mainEntityOfPage": URL[code], "inLanguage": LANG[code],
        "image": image, "datePublished": DATE, "dateModified": DATE,
    }.items():
        assert article[key] == value, (path, key)
    assert article["author"] == {"@id": person_id}, path
    assert article["encoding"] == {"@type": "MediaObject", "contentUrl": pdf, "encodingFormat": "application/pdf"}, path
    if code == "en":
        assert graph["WebSite"]["url"] == URL["en"], path

    flags = [a for a in page.anchors if "flag" in (a.get("class") or "").split()]
    others = [a for a in page.anchors if "flag" not in (a.get("class") or "").split()]
    assert len(flags) == 26 and all(a.get("target") in (None, "_self") for a in flags), path
    assert all(a.get("target") == "_blank" and {"noopener", "noreferrer"} <= set((a.get("rel") or "").split()) for a in others), path


assert sorted(ROOT.glob("*/index.html")) == sorted(ROOT / code / "index.html" for code in CODES if code != "en")
for language in CODES:
    check_page(language)

ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
entries = ET.parse(ROOT / "sitemap.xml").getroot().findall("s:url", ns)
assert len(entries) == 26
assert {entry.findtext("s:loc", namespaces=ns) for entry in entries} == set(URL.values())
assert all(entry.findtext("s:lastmod", namespaces=ns) == DATE for entry in entries)
assert all(entry.find("s:priority", ns) is None and entry.find("s:changefreq", ns) is None for entry in entries)
assert f"Sitemap: {DOMAIN}/sitemap.xml" in (ROOT / "robots.txt").read_text(encoding="utf-8")
assert (ROOT / f"{KEY}.txt").read_text(encoding="utf-8").strip() == KEY
print("SEO validation passed for all 26 pages, links, PDFs, sitemap, robots and IndexNow key.")
