"""NeuralMath — neural network from scratch with NumPy.

A compact, reproducible binary-classification example used by neuralmath.org.
The data are synthetic but kept within plausible mass/diameter ranges for the
teaching example. No machine-learning framework is used.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def relu(s):
    return np.maximum(0, s)


def relu_derivative(s):
    return (s > 0).astype(float)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def base_boundary(m):
    return (
        6.68
        + 0.74 * np.exp(-((m - 155) / 20) ** 2)
        + 1.50 / (1 + np.exp(-(m - 202) / 5))
    )


def build_dataset():
    masses_1 = np.linspace(118, 212, 16)
    masses_2 = masses_1 + np.array([
        1.2, -1.1, 0.8, -1.3, 1.0, -0.8, 1.1, -1.0,
        1.3, -0.9, 0.9, -1.2, 1.0, -1.0, 0.8, -0.7,
    ])
    k = np.arange(16)

    # Two apple bands above the reference boundary.
    apple_1 = np.column_stack([
        masses_1,
        base_boundary(masses_1) + 0.22 + 0.05 * np.sin(0.8 * k),
    ])
    apple_2 = np.column_stack([
        masses_2,
        base_boundary(masses_2) + 0.58 + 0.06 * np.cos(0.45 * k),
    ])

    # Two orange bands below the reference boundary.
    orange_1 = np.column_stack([
        masses_1,
        base_boundary(masses_1) - (0.22 + 0.04 * np.cos(0.7 * k)),
    ])
    orange_2 = np.column_stack([
        masses_2,
        base_boundary(masses_2) - (0.55 + 0.05 * np.sin(0.5 * k)),
    ])

    X_train = np.vstack([apple_1, apple_2, orange_1, orange_2])
    y_train = np.array([1.0] * 32 + [0.0] * 32)

    X_test = np.array([
        [121.0, 7.20], [135.0, 7.45], [152.0, 7.75], [171.0, 7.32],
        [121.0, 6.28], [152.0, 6.86], [189.0, 6.48], [208.0, 7.55],
    ])
    y_test = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    return X_train, y_train, X_test, y_test


def predict(X, mean, std, W1, b1, W2, b2):
    Xn = (X - mean) / std
    outputs = []
    for x in Xn:
        h = relu(W1 @ x + b1)
        y_hat = sigmoid(W2 @ h + b2)
        outputs.append(y_hat)
    return np.array(outputs)


def train_network(
    X_train,
    y_train,
    n_hidden=12,
    learning_rate=0.05,
    n_epochs=6000,
    seed=42,
):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    Xn = (X_train - mean) / std

    rng = np.random.default_rng(seed)
    W1 = 0.1 * rng.standard_normal((n_hidden, 2))
    b1 = np.zeros(n_hidden)
    W2 = 0.1 * rng.standard_normal(n_hidden)
    b2 = 0.0

    history = []
    for epoch in range(n_epochs):
        for i in rng.permutation(len(Xn)):
            x = Xn[i]
            target = y_train[i]

            # Forward pass.
            z1 = W1 @ x + b1
            h = relu(z1)
            z2 = W2 @ h + b2
            y_hat = sigmoid(z2)

            # Backpropagation for E = 1/2 (target - y_hat)^2.
            delta2 = (y_hat - target) * y_hat * (1 - y_hat)
            dW2 = delta2 * h
            db2 = delta2

            delta1 = (delta2 * W2) * relu_derivative(z1)
            dW1 = np.outer(delta1, x)
            db1 = delta1

            # Stochastic-gradient update.
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Mean loss with parameters fixed at the end of the epoch.
        errors = []
        for x, target in zip(Xn, y_train):
            h = relu(W1 @ x + b1)
            y_hat = sigmoid(W2 @ h + b2)
            errors.append(0.5 * (target - y_hat) ** 2)
        history.append(float(np.mean(errors)))

    return mean, std, W1, b1, W2, b2, history


def make_figure(
    X_train,
    y_train,
    X_test,
    y_test,
    mean,
    std,
    W1,
    b1,
    W2,
    b2,
    png_path,
    pdf_path,
):
    masses = np.linspace(116, 214, 500)
    diameters = np.linspace(6.0, 8.85, 500)
    M, D = np.meshgrid(masses, diameters)
    grid = np.column_stack([M.ravel(), D.ravel()])
    P = predict(grid, mean, std, W1, b1, W2, b2).reshape(M.shape)

    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.contourf(
        M,
        D,
        P,
        levels=[0.0, 0.25, 0.50, 0.75, 1.0],
        colors=["#f3e0cb", "#ecd5c4", "#dbe7cf", "#cde3bf"],
        alpha=0.95,
        zorder=0,
    )
    ax.contour(
        M,
        D,
        P,
        levels=[0.5],
        colors=["#5a1a8a"],
        linewidths=2.2,
        zorder=1,
    )

    mask = y_train == 1.0
    ax.scatter(
        X_train[mask, 0], X_train[mask, 1],
        marker="o", s=48, facecolors="white", edgecolors="black",
        linewidths=1.1, zorder=4, label="Apple — train",
    )
    mask = y_train == 0.0
    ax.scatter(
        X_train[mask, 0], X_train[mask, 1],
        marker="s", s=48, facecolors="white", edgecolors="black",
        linewidths=1.1, zorder=4, label="Orange — train",
    )
    mask = y_test == 1.0
    ax.scatter(
        X_test[mask, 0], X_test[mask, 1],
        marker="^", s=92, color="#1f77b4", zorder=5,
        label="Apple — test",
    )
    mask = y_test == 0.0
    ax.scatter(
        X_test[mask, 0], X_test[mask, 1],
        marker="x", s=110, color="#ff7f0e", linewidths=2.0, zorder=5,
        label="Orange — test",
    )

    ax.set_xlim(116, 214)
    ax.set_ylim(6.0, 8.85)
    ax.set_xlabel("Mass (g)")
    ax.set_ylabel("Diameter (cm)")
    ax.set_title("Neural network output after training", fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", markersize=10, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.1, linestyle="None",
               label="Apple — train"),
        Line2D([0], [0], marker="s", markersize=10, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.1, linestyle="None",
               label="Orange — train"),
        Line2D([0], [0], marker="^", markersize=10, color="#1f77b4",
               linestyle="None", label="Apple — test"),
        Line2D([0], [0], marker="x", markersize=11, color="#ff7f0e",
               markeredgewidth=2.0, linestyle="None", label="Orange — test"),
        Line2D([0], [0], color="#5a1a8a", lw=2.2,
               label="Neural-network decision boundary"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderpad=1.0,
        labelspacing=1.1,
        handlelength=2.8,
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    X_train, y_train, X_test, y_test = build_dataset()
    mean, std, W1, b1, W2, b2, history = train_network(X_train, y_train)

    pred_train = predict(X_train, mean, std, W1, b1, W2, b2)
    pred_test = predict(X_test, mean, std, W1, b1, W2, b2)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    make_figure(
        X_train, y_train, X_test, y_test,
        mean, std, W1, b1, W2, b2,
        output_dir / "neural_network_output_en.png",
        output_dir / "neural_network_output_en.pdf",
    )

    print(f"Final mean loss: {history[-1]:.10f}")
    print(f"Training accuracy: {np.mean((pred_train >= 0.5) == y_train):.4f}")
    print(f"Illustrative test accuracy: {np.mean((pred_test >= 0.5) == y_test):.4f}")
    print("Test outputs:", np.round(pred_test, 6))


if __name__ == "__main__":
    main()
