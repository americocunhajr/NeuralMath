# NeuralMath

**NeuralMath** contains the companion Python codes for the manuscript

> **A. Cunha Jr, _A anatomia matemática de uma rede neural_, manuscript prepared for submission to Professor de Matemática Online (PMO), 2026.**

The repository implements, from scratch and using only NumPy, the small neural network developed mathematically in the article. The example is deliberately educational: two input variables (fruit mass and diameter), one hidden layer with three ReLU neurons, one sigmoid output, quadratic loss, backpropagation, and stochastic gradient descent.

<p align="center">
<img src="output/neural_network_output.png" width="80%">
</p>

<p align="center">
<a href="https://colab.research.google.com/github/americocunhajr/NeuralMath/blob/main/ColabCodes/NeuralMath_Colab.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory organization](#directory-organization)
- [Quick Start](#quick-start)
- [Google Colab](#google-colab)
- [Mathematical model](#mathematical-model)
- [Reproducibility](#reproducibility)
- [Output](#output)
- [Author](#author)
- [Citation](#citation)
- [License](#license)

## Overview

The purpose of **NeuralMath** is to make the connection between the equations of a neural network and their computational implementation explicit. No machine-learning framework is used. Every step appearing in the code corresponds directly to an operation developed in the manuscript:

1. standardization of the inputs;
2. affine transformation in the hidden layer;
3. ReLU activation;
4. affine transformation in the output layer;
5. sigmoid activation;
6. quadratic loss;
7. derivatives obtained by the chain rule;
8. backpropagation;
9. stochastic gradient-descent updates.

The training dataset has an XOR-like geometry and is not linearly separable. This makes the hidden nonlinear layer genuinely necessary for the classification task.

## Features

- Neural network implemented from scratch with NumPy
- Two inputs: mass and diameter
- Three-neuron hidden layer with ReLU activation
- Sigmoid output for binary classification
- Explicit backpropagation formulas
- Stochastic gradient descent with shuffled examples
- Training/test separation
- Reproducible random seed
- Automatic generation of the decision-region figure
- Ready-to-run Google Colab notebook
- No TensorFlow, PyTorch, or scikit-learn required

## Directory organization

```text
NeuralMath/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── _config.yml
├── neural_network_from_scratch.py
├── NeuralMath_Colab.ipynb
└── output/
    ├── neural_network_output.png
    ├── neural_network_output.pdf
    └── training_results.txt
```

## Quick Start

Get a local copy of **NeuralMath**:

```bash
git clone https://github.com/americocunhajr/NeuralMath.git
cd NeuralMath
```

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
```

Run the complete example:

```bash
python neural_network_from_scratch.py
```

The numerical output is printed to the terminal and the files are written to `output/`.

## Google Colab

The full example can be executed in the browser without any local installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/americocunhajr/NeuralMath/blob/main/NeuralMath_Colab.ipynb)

Open the notebook and select **Runtime → Run all**.

## Mathematical model

For an input vector

\[
\mathbf{x}
=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix},
\]

the hidden layer computes

\[
\mathbf{z}^{(1)}
=
W^{(1)}\mathbf{x}
+
\mathbf{b}^{(1)},
\qquad
\mathbf{h}
=
\operatorname{ReLU}
\left(
\mathbf{z}^{(1)}
\right).
\]

The output neuron computes

\[
z^{(2)}
=
W^{(2)}\mathbf{h}
+
b^{(2)},
\qquad
\widehat{y}
=
\sigma
\left(
z^{(2)}
\right).
\]

The quadratic loss for one example is

\[
E
=
\frac{1}{2}
\left(
y-\widehat{y}
\right)^2.
\]

The parameters are updated according to

\[
\theta
\leftarrow
\theta
-
\eta
\frac{\partial E}{\partial\theta}.
\]

All derivatives used by the program are written explicitly in `neural_network_from_scratch.py` and in the Colab notebook.

## Reproducibility

The code uses

```python
rng = np.random.default_rng(42)
```

and therefore reproduces the same initialization, shuffled training sequence, final predictions, and decision regions for a fixed NumPy environment.

The mean loss is recomputed with fixed network parameters at the end of every epoch, so the reported quantity corresponds to

\[
J
=
\frac{1}{N}
\sum_{i=1}^{N}
E_i.
\]

## Output

A typical run begins with a mean loss near

```text
epoca    0  perda_media = 0.125130103
```

and finishes after 2000 epochs near

```text
epoca 1999  perda_media = 0.000275855
```

All eight training examples and the four illustrative test examples are classified according to their prescribed labels with the fixed seed used in the repository.

The file `output/neural_network_output.png` shows the nonlinear decision regions learned by the network in the original physical variables, mass and diameter.

## Author

**Americo Cunha Jr**  
Laboratório Nacional de Computação Científica (LNCC), Petrópolis, Brazil  
Universidade do Estado do Rio de Janeiro (UERJ), Rio de Janeiro, Brazil  
<http://americocunha.org>

## Citation

If these codes are used in teaching, research, or derivative work, please cite the associated manuscript:

> **A. Cunha Jr**, _A anatomia matemática de uma rede neural_, manuscript prepared for submission to Professor de Matemática Online (PMO), 2026.

```bibtex
@article{CunhaJr2026NeuralMath,
  author  = {A. {Cunha~Jr}},
  title   = {A anatomia matem{\'a}tica de uma rede neural},
  journal = {Professor de Matem{\'a}tica Online},
  year    = {2026},
  note    = {Manuscript prepared for submission},
  url     = {https://github.com/americocunhajr/NeuralMath}
}
```

## License

**NeuralMath** is released under the MIT license. See the `LICENSE` file for details.

Contributions are welcome and are distributed under the same license.
