"""
NeuralMath
==========

Implementação didática, do zero, de uma pequena rede neural usando apenas NumPy.

Este exemplo acompanha o artigo:
"A anatomia matemática de uma rede neural", Americo Cunha Jr.

O objetivo é construir um conjunto de dados plausível para maçãs e laranjas
usando apenas duas variáveis físicas:

    x1 = massa (g)
    x2 = diâmetro (cm)

Para obter uma fronteira de decisão mais sinuosa e, ao mesmo tempo, evitar um
conjunto excessivamente artificial, o programa distribui 64 exemplos de treino
por uma região ampla do plano massa--diâmetro. Os dados são sintéticos, mas
permanecem em faixas plausíveis para as duas frutas. O conjunto de teste possui
8 exemplos adicionais.

A rede utiliza:
    - 2 entradas;
    - 1 camada oculta com 12 neurônios ReLU;
    - 1 neurônio de saída com função sigmoide;
    - erro quadrático;
    - retropropagação;
    - descida do gradiente estocástica.

Ao executar o arquivo, o programa gera:
    output/neural_network_output_pt.png
    output/neural_network_output_pt.pdf
    output/training_results_pt.txt
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def relu(s):
    return np.maximum(0, s)


def relu_derivada(s):
    return (s > 0).astype(float)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def fronteira_base(m):
    return 6.68 + 0.74 * np.exp(-((m - 155) / 20) ** 2) + 1.50 / (1 + np.exp(-(m - 202) / 5))


def construir_dados():
    massas_1 = np.linspace(118, 212, 16)
    massas_2 = massas_1 + np.array([1.2, -1.1, 0.8, -1.3, 1.0, -0.8, 1.1, -1.0,
                                     1.3, -0.9, 0.9, -1.2, 1.0, -1.0, 0.8, -0.7])
    k = np.arange(16)

    # Maçãs: duas faixas acima da fronteira base
    a1 = np.column_stack([
        massas_1,
        fronteira_base(massas_1) + 0.22 + 0.05 * np.sin(0.8 * k)
    ])
    a2 = np.column_stack([
        massas_2,
        fronteira_base(massas_2) + 0.58 + 0.06 * np.cos(0.45 * k)
    ])

    # Laranjas: duas faixas abaixo da fronteira base
    l1 = np.column_stack([
        massas_1,
        fronteira_base(massas_1) - (0.22 + 0.04 * np.cos(0.7 * k))
    ])
    l2 = np.column_stack([
        massas_2,
        fronteira_base(massas_2) - (0.55 + 0.05 * np.sin(0.5 * k))
    ])

    X_treino = np.vstack([a1, a2, l1, l2])
    y_treino = np.array([1.0] * 32 + [0.0] * 32)

    X_teste = np.array([
        [121.0, 7.20], [135.0, 7.45], [152.0, 7.75], [171.0, 7.32],
        [121.0, 6.28], [152.0, 6.86], [189.0, 6.48], [208.0, 7.55]
    ])
    y_teste = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    return X_treino, y_treino, X_teste, y_teste


def prever(X, media, desvio, W1, b1, W2, b2):
    Xn = (X - media) / desvio
    saidas = []
    for x in Xn:
        h = relu(W1 @ x + b1)
        y_hat = sigmoid(W2 @ h + b2)
        saidas.append(y_hat)
    return np.array(saidas)


def treinar_rede(X_treino, y_treino, n_ocultos=12, eta=0.05, n_epocas=6000, semente=42):
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    Xn = (X_treino - media) / desvio

    rng = np.random.default_rng(semente)
    W1 = 0.1 * rng.standard_normal((n_ocultos, 2))
    b1 = np.zeros(n_ocultos)
    W2 = 0.1 * rng.standard_normal(n_ocultos)
    b2 = 0.0

    historico = []
    for epoca in range(n_epocas):
        for i in rng.permutation(len(Xn)):
            x = Xn[i]
            alvo = y_treino[i]

            z1 = W1 @ x + b1
            h = relu(z1)
            z2 = W2 @ h + b2
            y_hat = sigmoid(z2)

            delta2 = (y_hat - alvo) * y_hat * (1 - y_hat)
            dW2 = delta2 * h
            db2 = delta2

            delta1 = (delta2 * W2) * relu_derivada(z1)
            dW1 = np.outer(delta1, x)
            db1 = delta1

            W1 -= eta * dW1
            b1 -= eta * db1
            W2 -= eta * dW2
            b2 -= eta * db2

        erros = []
        for x, alvo in zip(Xn, y_treino):
            h = relu(W1 @ x + b1)
            y_hat = sigmoid(W2 @ h + b2)
            erros.append(0.5 * (alvo - y_hat) ** 2)
        historico.append(float(np.mean(erros)))

    return media, desvio, W1, b1, W2, b2, historico


def gerar_figura(X_treino, y_treino, X_teste, y_teste, media, desvio, W1, b1, W2, b2, caminho_png, caminho_pdf):
    massas = np.linspace(116, 214, 500)
    diametros = np.linspace(6.0, 8.85, 500)
    M, D = np.meshgrid(massas, diametros)
    grade = np.column_stack([M.ravel(), D.ravel()])
    P = prever(grade, media, desvio, W1, b1, W2, b2).reshape(M.shape)

    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.contourf(
        M, D, P,
        levels=[0.0, 0.25, 0.50, 0.75, 1.0],
        colors=["#f3e0cb", "#ecd5c4", "#dbe7cf", "#cde3bf"],
        alpha=0.95,
        zorder=0,
    )
    ax.contour(M, D, P, levels=[0.5], colors=["#5a1a8a"], linewidths=2.2, zorder=1)

    mask = y_treino == 1.0
    ax.scatter(
        X_treino[mask, 0], X_treino[mask, 1], marker="o", s=48,
        facecolors="white", edgecolors="black", linewidths=1.1,
        zorder=4, label="Maçã — treino"
    )
    mask = y_treino == 0.0
    ax.scatter(
        X_treino[mask, 0], X_treino[mask, 1], marker="s", s=48,
        facecolors="white", edgecolors="black", linewidths=1.1,
        zorder=4, label="Laranja — treino"
    )
    mask = y_teste == 1.0
    ax.scatter(
        X_teste[mask, 0], X_teste[mask, 1], marker="^", s=92,
        color="#1f77b4", zorder=5, label="Maçã — teste"
    )
    mask = y_teste == 0.0
    ax.scatter(
        X_teste[mask, 0], X_teste[mask, 1], marker="x", s=110,
        color="#ff7f0e", linewidths=2.0, zorder=5, label="Laranja — teste"
    )

    ax.set_xlim(116, 214)
    ax.set_ylim(6.0, 8.85)
    ax.set_xlabel("Massa (g)")
    ax.set_ylabel("Diâmetro (cm)")
    ax.set_title("Saída da rede neural após o treinamento", fontweight="bold")

    handles = [
        Line2D([0], [0], marker='o', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.1, linestyle='None', label='Maçã — treino'),
        Line2D([0], [0], marker='s', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.1, linestyle='None', label='Laranja — treino'),
        Line2D([0], [0], marker='^', markersize=10, color='#1f77b4', linestyle='None', label='Maçã — teste'),
        Line2D([0], [0], marker='x', markersize=11, color='#ff7f0e', markeredgewidth=2.0, linestyle='None', label='Laranja — teste'),
        Line2D([0], [0], color='#5a1a8a', lw=2.2, label='Fronteira de decisão da rede neural'),
    ]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, borderpad=1.0, labelspacing=1.1, handlelength=2.8)

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=220, bbox_inches='tight')
    fig.savefig(caminho_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    X_treino, y_treino, X_teste, y_teste = construir_dados()
    media, desvio, W1, b1, W2, b2, historico = treinar_rede(X_treino, y_treino)

    pred_treino = prever(X_treino, media, desvio, W1, b1, W2, b2)
    pred_teste = prever(X_teste, media, desvio, W1, b1, W2, b2)

    output_dir = Path(__file__).resolve().parent / 'output'
    output_dir.mkdir(exist_ok=True)

    gerar_figura(
        X_treino, y_treino, X_teste, y_teste,
        media, desvio, W1, b1, W2, b2,
        output_dir / 'neural_network_output_pt.png',
        output_dir / 'neural_network_output_pt.pdf',
    )

    linhas = []
    linhas.append('NeuralMath — versão em português')
    linhas.append(f'Perda média final: {historico[-1]:.10f}')
    linhas.append(f'Acurácia no treinamento: {np.mean((pred_treino >= 0.5) == y_treino):.4f}')
    linhas.append(f'Acurácia no teste ilustrativo: {np.mean((pred_teste >= 0.5) == y_teste):.4f}')
    linhas.append('')
    linhas.append('Saídas no teste:')
    for fruta, alvo, y_hat in zip(X_teste, y_teste, pred_teste):
        prevista = 'maçã' if y_hat >= 0.5 else 'laranja'
        correta = 'maçã' if alvo == 1 else 'laranja'
        linhas.append(f'{fruta} -> saida={y_hat:.6f}; prevista={prevista}; correta={correta}')

    texto = '\n'.join(linhas)
    (output_dir / 'training_results_pt.txt').write_text(texto, encoding='utf-8')
    print(texto)


if __name__ == '__main__':
    main()
