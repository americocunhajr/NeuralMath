# NeuralMath multilingual website — validation report

Generated: 2026-10-06

## Scope

- 26 language editions: PT, EN, ES, FR, IT, DE, AR, BN, CS, EL, FA, HI, HU, ID, JA, KO, NL, PL, RO, RU, SV, TR, UK, UR, VI and ZH.
- 8 core pedagogical/theory sections plus the manuscript references in every language edition.
- Figures 1–7 in every language (182 localized PNG figures total).
- One dedicated Google Colab notebook per language.
- Portuguese and English remain first and second in language navigation; ES, FR, IT and DE retain their previous order before the new editions.
- Every page cites the original Portuguese pedagogical text: *A anatomia matemática de uma rede neural*.

## Colab overflow correction

The long notebook file path remains only inside the Colab URL. It is no longer displayed as card text. Resource cards use short labels such as `Notebook · Português`, while CSS includes `min-width: 0`, `overflow-wrap: anywhere` and `word-break: break-word` as defensive safeguards.

## Colab localization

- Each edition links to `ColabCodes/NeuralMath_Colab_XX.ipynb` for its language.
- Portuguese keeps Portuguese pedagogical code naming.
- Every non-Portuguese notebook keeps Python variables and function identifiers in English.
- Instructional Markdown is localized to the language of the page.
- In the 20 newly added notebooks, plot title, axes and legend text are taken directly from the corresponding translated Figure 7 source; numerical output uses language-neutral mathematical/program identifiers (`E_final`, `acc_train`, `acc_test`, `y_hat_test`).

## Structural checks

- 26 HTML language pages found.
- 9 manuscript-content blocks on every page: 8 core sections + references.
- 7 localized figures on every page.
- 26 language-navigation entries on every page.
- PDF, GitHub and language-specific Colab links present at the top and near the bottom of every page.
- No visible Colab link text contains `.ipynb` or the long notebook path.
- All local `href` and `src` targets resolve.
- 26 dedicated notebooks parse as valid Jupyter JSON.
- Non-Portuguese notebooks retain English Python identifiers.
- RTL layout enabled for Arabic, Persian and Urdu.
- Sitemap updated for all 26 language URLs.

## Execution check

The updated Indonesian and Japanese notebooks were executed end-to-end using the current 2–12–1 experiment. Both reproduce:

- final mean loss: `0.0001785644`
- training accuracy: `1.0000`
- illustrative test accuracy: `1.0000`

The local Japanese execution environment reports missing CJK glyphs for Matplotlib's default DejaVu Sans font; this does not affect computation or the website figures, which are the supplied localized figure assets.
