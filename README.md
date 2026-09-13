# NeuralMath

**The mathematics behind neural networks — from equations to executable code.**

[![Website](https://img.shields.io/badge/website-neuralmath.org-53d7ff)](https://neuralmath.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-NumPy-7b8cff)](https://www.python.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8342--0363-A6CE39)](https://orcid.org/0000-0002-8342-0363)

**NeuralMath** is an independent educational project that develops a transparent,
mathematically grounded introduction to neural networks. Its central idea is
simple: a neural network should not be introduced as a black box. The equations,
the geometry, the learning algorithm, and the Python implementation should be
read together.

The project combines a concise web narrative, a longer pedagogical text in PDF,
reproducible Python/NumPy code, notebooks, figures, and multilingual versions of
the website.

**Website:** https://neuralmath.org  
**Repository:** https://github.com/americocunhajr/NeuralMath

---

## Citation

If NeuralMath is useful in teaching, research, course material, talks, or
derivative work, please **cite the pedagogical text rather than the website**.

> **Americo Cunha Jr**, *A anatomia matemática de uma rede neural*, manuscript submitted to **Professor de Matemática Online (PMO)**, 2026.

```bibtex
@article{CunhaJr2026NeuralMath,
  author  = {A. {Cunha~Jr}},
  title   = {A anatomia matem{\'a}tica de uma rede neural},
  journal = {Professor de Matem{\'a}tica Online},
  year    = {2026},
  note    = {Manuscript submitted for publication},
 url      = {http://neuralmth.org},
}
```

The bibliographic entry above is provisional. It should be replaced by the
published PMO reference, including DOI, volume, number, and pages, as soon as
those data become available.

---

## What this repository contains

NeuralMath is organized around a complete neural-network example implemented
**from scratch with NumPy**. No automatic differentiation and no high-level
machine-learning framework are required to understand the training procedure.

The current computational experiment uses:

- two input variables: fruit mass and diameter;
- 64 synthetic training examples and 8 illustrative test examples;
- one hidden layer with 12 ReLU neurons;
- one sigmoid output neuron;
- 49 trainable parameters;
- quadratic loss;
- explicit backpropagation via the chain rule;
- stochastic gradient descent with shuffled examples;
- a fixed random seed for reproducibility;
- a two-dimensional decision-region visualization.

The data are synthetic and pedagogical, but they are distributed over plausible
mass–diameter ranges. The geometry is intentionally nonlinear so that the hidden
layer is genuinely useful.

<p align="center">
  <img src="output/neural_network_output_en.png" alt="Decision regions learned by the NeuralMath example" width="82%">
</p>

---

## From geometry to a neural network

The pedagogical path begins with a two-dimensional classification problem.
A fruit is represented by

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix},
$$

where $x_1$ is mass and $x_2$ is diameter.

A linear classifier evaluates

$$
s = w_1x_1 + w_2x_2 + b.
$$

This affine expression is the mathematical core of an artificial neuron. A
nonlinear activation then transforms the score,

$$
\widehat{y} = g(s).
$$

By connecting several neurons, the model becomes a composition of functions.
For the computational example,

$$
\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)},
\qquad
\mathbf{h} = \operatorname{ReLU}(\mathbf{z}^{(1)}),
$$

followed by

$$
z^{(2)} = W^{(2)}\mathbf{h} + b^{(2)},
\qquad
\widehat{y}=\sigma(z^{(2)}).
$$

Training adjusts the parameters so as to reduce a loss. For one example,

$$
E = \frac{1}{2}(y-\widehat{y})^2,
$$

and the parameters are updated according to

$$
\theta \leftarrow
\theta - \eta\frac{\partial E}{\partial\theta}.
$$

The repository implements these derivatives explicitly so that the code mirrors
the mathematics.

---

## Repository structure

The canonical project is intended to follow this organization:

```text
NeuralMath/
├── README.md
├── LICENSE.txt
├── CITATION.cff
├── requirements.txt
│
├── rede_neural_do_zero.py
├── neural_network_from_scratch.py
├── NeuralMath_Colab_PT_BR.ipynb
├── NeuralMath_Colab_EN.ipynb
│
├── output/
│   ├── neural_network_output_pt.png
│   ├── neural_network_output_pt.pdf
│   ├── neural_network_output_en.png
│   ├── neural_network_output_en.pdf
│   ├── training_results_pt.txt
│   └── training_results_en.txt
│
├── paper/
│   └── [pedagogical PDF / preprint]
│
└── docs/
    ├── index.html
    ├── pt/
    ├── es/
    ├── fr/
    ├── it/
    ├── de/
    ├── assets/
    └── pdf/
```

The `docs/` directory is used by GitHub Pages to publish
**https://neuralmath.org**.

---

## Website and languages

The website is designed as a **single continuous pedagogical page per language**.
Instead of separating theory and code into independent sections of the site, the
reader progresses through a compact narrative:

**problem → geometry → neuron → activation → network → loss → gradients →
backpropagation → Python → learned decision boundary**

Longer derivations and the complete pedagogical treatment remain available as
PDF documents.

The current website structure supports:

- English;
- Portuguese;
- Spanish;
- French;
- Italian;
- German.

The mathematical model and computational experiment are the same in every
language; only the exposition is translated.

---

## Running the Python example

Clone the repository:

```bash
git clone https://github.com/americocunhajr/NeuralMath.git
cd NeuralMath
```

Create a virtual environment if desired:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the English version:

```bash
python neural_network_from_scratch.py
```

or the Portuguese version:

```bash
python rede_neural_do_zero.py
```

The scripts train the network, print the numerical results, and regenerate the
decision-region figure.

---

## Google Colab

The notebooks provide the same experiment in a browser-based environment:

- **English:**  
  https://colab.research.google.com/github/americocunhajr/NeuralMath/blob/main/NeuralMath_Colab_EN.ipynb

- **Português:**  
  https://colab.research.google.com/github/americocunhajr/NeuralMath/blob/main/NeuralMath_Colab_PT_BR.ipynb

---

## Reproducibility

The reference implementation uses a fixed NumPy random generator,

```python
rng = np.random.default_rng(42)
```

and keeps the complete training procedure explicit. Training-set statistics are
used for standardization, and the same deterministic setup is used when the
reference figures are generated.

The project is intended to make numerical experiments inspectable and
reproducible rather than to present the example as a benchmark for statistical
generalization.

---

## Future extensions

The current binary-classification example is the first module of a broader
project. Natural extensions include:

- visualization of training dynamics;
- interactive inspection of hidden neurons;
- gradient checking;
- alternative activation and loss functions;
- multiclass classification and softmax;
- deeper networks;
- optimization methods beyond basic SGD;
- browser-based interactive experiments;
- further pedagogical texts and multilingual editions.

The goal is to preserve the same principle throughout: **the mathematics and the
code should remain close enough that the reader can trace one directly into the
other.**

---

## Author

**Americo Cunha Jr** is a computational scientist working at the interface of
nonlinear dynamics, uncertainty, data-driven modeling, and artificial
intelligence. He completed his doctoral training in Mechanical Engineering at
**PUC-Rio** and **Université Paris-Est**.

NeuralMath is a **personal and independent educational project**.

- Personal page: https://americocunha.org
- ORCID: https://orcid.org/0000-0002-8342-0363
- NeuralMath: https://neuralmath.org

---

## License

Except where explicitly stated otherwise, the original educational content of
NeuralMath is released under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

<a href="https://creativecommons.org/licenses/by/4.0/">
  <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png"
       alt="Creative Commons Attribution 4.0 International"
       width="88">
</a>

You may share and adapt the material, including for commercial purposes, as long
as appropriate attribution is provided.

See [`LICENSE.txt`](LICENSE.txt) for the repository-wide licensing notice and
important exceptions. In particular, journal-formatted PDFs or third-party
materials may be governed by separate terms.

---

## Acknowledgment

If you use NeuralMath in a course, lecture, workshop, research project, or
educational resource, citation of the associated pedagogical text is appreciated.
Feedback, corrections, translations, and mathematically motivated extensions are
also welcome.
