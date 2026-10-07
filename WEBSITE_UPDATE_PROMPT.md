# NeuralMath website maintenance prompt

Maintain the existing dark NeuralMath visual identity and treat the website as a rich multilingual web edition of the pedagogical article.

## Languages

The website contains 26 languages. Keep Brazil first and English second in the flag navigation. Preserve the existing first six order (PT, EN, ES, FR, IT, DE); append the additional languages in the current repository order. Every language must have:

- a complete theory page based on its translated manuscript;
- the translated article PDF;
- Figures 1–7 in that language;
- a dedicated Colab notebook;
- links to PDF, GitHub, and Colab at the top and near the bottom;
- hreflang, canonical, accessible language navigation, and sitemap entry.

## Colab notebooks

Portuguese may use Portuguese identifiers. Every other notebook must use English variable and function names. Explanatory Markdown must be in the language of the page.

Never display the raw long notebook path as resource-card copy. The card may link to the full URL but should display a short label such as `Notebook · Português`. Keep CSS safeguards (`min-width:0` and `overflow-wrap:anywhere`) so long URLs or filenames cannot overflow cards.

## Citation

Every language must cite the same original Portuguese pedagogical text:

Americo Cunha Jr. “A anatomia matemática de uma rede neural.” Manuscript submitted to Professor de Matemática Online (PMO), 2026.

```bibtex
@unpublished{cunha2026anatomia,
  author = {Cunha Jr, Americo},
  title  = {A anatomia matemática de uma rede neural},
  note   = {Manuscript submitted to Professor de Matemática Online (PMO)},
  year   = {2026}
}
```

State that the bibliographic entry will be updated after publication and that readers should cite the pedagogical text rather than the website.

## Validation

Before release, verify all local href/src paths, 8 theory sections per language, 7 figures per language, a PDF and a Colab notebook per language, language order, mobile wrapping, RTL rendering for Arabic/Persian/Urdu, and that no Colab filename overflows its resource card.
