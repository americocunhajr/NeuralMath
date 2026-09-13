"""
NeuralMath
==========

Minimal neural network implemented from scratch with NumPy.

Companion code for the manuscript:
"A anatomia matemática de uma rede neural", A. Cunha Jr (2026).

The example uses a deliberately non-linearly separable fruit dataset,
a hidden layer with three ReLU neurons, a sigmoid output, quadratic loss,
backpropagation, and stochastic gradient descent.

Running this file creates:
    output/neural_network_output.png
    output/neural_network_output.pdf
    output/training_results.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def relu(s):
    """Rectified Linear Unit."""
    return np.maximum(0, s)


def relu_derivada(s):
    """Derivative convention used for ReLU; at zero we take 0."""
    return (s > 0).astype(float)


def sigmoid(s):
    """Logistic sigmoid."""
    return 1 / (1 + np.exp(-s))


def prever(Xn, W1, b1, W2, b2):
    """Return network outputs for normalized input rows."""
    saidas = []

    for x in Xn:
        h = relu(W1 @ x + b1)
        y_hat = sigmoid(W2 @ h + b2)
        saidas.append(y_hat)

    return np.array(saidas)


def main():
    # ----------------------------------------------------------
    # Data: [mass (g), diameter (cm)]
    # The geometry is XOR-like and cannot be separated by one line.
    # ----------------------------------------------------------
    X_treino = np.array([
        [130.0, 6.6],
        [134.0, 6.5],
        [170.0, 7.8],
        [174.0, 7.7],
        [130.0, 7.8],
        [126.0, 7.7],
        [170.0, 6.6],
        [166.0, 6.5],
    ])

    # 1 = apple; 0 = orange
    y_treino = np.array([
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
    ])

    # Examples not used to adjust the parameters
    X_teste = np.array([
        [132.0, 6.55],
        [172.0, 7.75],
        [128.0, 7.75],
        [168.0, 6.55],
    ])

    y_teste = np.array([1.0, 1.0, 0.0, 0.0])

    # ----------------------------------------------------------
    # Standardization using training-set statistics only
    # ----------------------------------------------------------
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)

    Xn = (X_treino - media) / desvio
    Xn_teste = (X_teste - media) / desvio

    # ----------------------------------------------------------
    # Initialize the 13 parameters
    # ----------------------------------------------------------
    rng = np.random.default_rng(42)

    W1 = 0.1 * rng.standard_normal((3, 2))
    b1 = np.zeros(3)

    W2 = 0.1 * rng.standard_normal(3)
    b2 = 0.0

    eta = 0.1
    n_epocas = 2000
    historico = []

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    for epoca in range(n_epocas):

        # Random order -> elementary stochastic gradient descent
        ordem = rng.permutation(len(Xn))

        for i in ordem:

            x = Xn[i]
            alvo = y_treino[i]

            # Forward propagation
            z1 = W1 @ x + b1
            h = relu(z1)

            z2 = W2 @ h + b2
            y_hat = sigmoid(z2)

            # Backpropagation
            delta2 = (
                (y_hat - alvo)
                * y_hat
                * (1 - y_hat)
            )

            dW2 = delta2 * h
            db2 = delta2

            delta1 = (
                (delta2 * W2)
                * relu_derivada(z1)
            )

            dW1 = np.outer(delta1, x)
            db1 = delta1

            # Gradient descent
            W1 -= eta * dW1
            b1 -= eta * db1
            W2 -= eta * dW2
            b2 -= eta * db2

        # Mean loss at fixed parameters at the end of the epoch
        erros = []

        for x, alvo in zip(Xn, y_treino):
            h = relu(W1 @ x + b1)
            y_hat = sigmoid(W2 @ h + b2)
            E = 0.5 * (alvo - y_hat) ** 2
            erros.append(E)

        erro_medio = np.mean(erros)
        historico.append(float(erro_medio))

        if epoca % 200 == 0:
            print(
                f"epoca {epoca:4d}  "
                f"perda_media = {erro_medio:.9f}"
            )

    print(
        f"epoca {n_epocas - 1:4d}  "
        f"perda_media = {historico[-1]:.9f}"
    )

    # ----------------------------------------------------------
    # Final predictions
    # ----------------------------------------------------------
    pred_treino = prever(Xn, W1, b1, W2, b2)
    pred_teste = prever(Xn_teste, W1, b1, W2, b2)

    print("\nTreinamento:")

    for fruta, alvo, y_hat in zip(
            X_treino, y_treino, pred_treino):

        prevista = (
            "maca" if y_hat >= 0.5
            else "laranja"
        )

        correta = (
            "maca" if alvo == 1
            else "laranja"
        )

        print(
            fruta,
            f"saida = {y_hat:.8f}",
            f"prevista = {prevista}",
            f"correta = {correta}",
        )

    print("\nTeste:")

    for fruta, alvo, y_hat in zip(
            X_teste, y_teste, pred_teste):

        prevista = (
            "maca" if y_hat >= 0.5
            else "laranja"
        )

        correta = (
            "maca" if alvo == 1
            else "laranja"
        )

        print(
            fruta,
            f"saida = {y_hat:.8f}",
            f"prevista = {prevista}",
            f"correta = {correta}",
        )

    # ----------------------------------------------------------
    # Figure: learned decision regions in the original units
    # ----------------------------------------------------------
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    massas = np.linspace(120, 180, 350)
    diametros = np.linspace(6.3, 8.0, 350)
    M, D = np.meshgrid(massas, diametros)

    grade = np.column_stack([M.ravel(), D.ravel()])
    grade_n = (grade - media) / desvio

    P = prever(
        grade_n, W1, b1, W2, b2
    ).reshape(M.shape)

    fig, ax = plt.subplots(figsize=(8.2, 5.7))

    ax.contourf(
        M, D, P,
        levels=[0.0, 0.5, 1.0],
        alpha=0.25,
    )

    ax.contour(
        M, D, P,
        levels=[0.5],
        linewidths=2,
    )

    for classe, rotulo in [
        (1.0, "Maçã — treino"),
        (0.0, "Laranja — treino"),
    ]:
        mask = y_treino == classe
        ax.scatter(
            X_treino[mask, 0],
            X_treino[mask, 1],
            s=70,
            label=rotulo,
        )

    for classe, rotulo in [
        (1.0, "Maçã — teste"),
        (0.0, "Laranja — teste"),
    ]:
        mask = y_teste == classe
        ax.scatter(
            X_teste[mask, 0],
            X_teste[mask, 1],
            marker="x",
            s=85,
            linewidths=2,
            label=rotulo,
        )

    ax.set_xlabel("Massa (g)")
    ax.set_ylabel("Diâmetro (cm)")
    ax.set_title(
        "Saída da rede neural após o treinamento"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "neural_network_output.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "neural_network_output.pdf",
        bbox_inches="tight",
    )

    # ----------------------------------------------------------
    # Save a text copy of the main numerical results
    # ----------------------------------------------------------
    linhas = []

    for i in range(0, n_epocas, 200):
        linhas.append(
            f"epoca {i:4d}  "
            f"perda_media = {historico[i]:.9f}"
        )

    linhas.append(
        f"epoca {n_epocas - 1:4d}  "
        f"perda_media = {historico[-1]:.9f}"
    )

    linhas.append("\nTreinamento:")

    for fruta, alvo, y_hat in zip(
            X_treino, y_treino, pred_treino):

        prevista = (
            "maca" if y_hat >= 0.5
            else "laranja"
        )

        correta = (
            "maca" if alvo == 1
            else "laranja"
        )

        linhas.append(
            f"{fruta.tolist()}  "
            f"saida = {y_hat:.8f}  "
            f"prevista = {prevista:7s}  "
            f"correta = {correta}"
        )

    linhas.append("\nTeste:")

    for fruta, alvo, y_hat in zip(
            X_teste, y_teste, pred_teste):

        prevista = (
            "maca" if y_hat >= 0.5
            else "laranja"
        )

        correta = (
            "maca" if alvo == 1
            else "laranja"
        )

        linhas.append(
            f"{fruta.tolist()}  "
            f"saida = {y_hat:.8f}  "
            f"prevista = {prevista:7s}  "
            f"correta = {correta}"
        )

    (output_dir / "training_results.txt").write_text(
        "\n".join(linhas) + "\n",
        encoding="utf-8",
    )

    plt.show()


if __name__ == "__main__":
    main()
