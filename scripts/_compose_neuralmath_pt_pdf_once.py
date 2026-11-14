"""Recompõe o PDF da coleção NeuralMath com a capa editorial original
e o texto matemático revisado para a PMO.
Uso temporário durante a sincronização de novembro de 2026.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
from pypdf import PdfReader, PdfWriter

root = Path(__file__).resolve().parents[1]
cover_path = root / "docs/pdf/NeuralMath_PT.pdf"
body_path = root / "docs/pdf/Anatomia_Matematica_PMO_Revisado_2026.pdf"

old = PdfReader(str(cover_path), strict=True)
body = PdfReader(str(body_path), strict=True)

assert len(old.pages) == 26, f"Original NeuralMath inesperado: {len(old.pages)} páginas"
assert len(body.pages) == 24, f"Revisão PMO inesperada: {len(body.pages)} páginas"
assert "NeuralMath" in (old.pages[0].extract_text() or ""), "Capa NeuralMath não encontrada"
assert "Resumo" in (body.pages[0].extract_text() or ""), "Manuscrito revisado não encontrado"

new = PdfWriter()
# Preserva literalmente as quatro páginas preliminares da edição online:
# capa NeuralMath, ficha, prefácio e informações da coleção.
for page in old.pages[:4]:
    new.add_page(page)
# Substitui somente o corpo antigo pelo texto final da PMO.
for page in body.pages:
    new.add_page(page)
new.add_metadata({
    "/Title": "A anatomia matemática de uma rede neural — coleção NeuralMath",
    "/Author": "Americo Cunha Jr.",
    "/Subject": "Edição portuguesa com manuscrito revisado em novembro de 2026",
})
assert len(new.pages) == 28
with NamedTemporaryFile(mode="wb", dir=cover_path.parent, suffix=".pdf", delete=False) as out:
    temp = Path(out.name)
    new.write(out)
check = PdfReader(str(temp), strict=True)
assert len(check.pages) == 28
assert "NeuralMath" in (check.pages[0].extract_text() or "")
assert "entropia cruzada" in " ".join((p.extract_text() or "").lower() for p in check.pages[4:16])
os.replace(temp, cover_path)
print(f"PDF NeuralMath recomposto: {cover_path}, {len(check.pages)} páginas")
