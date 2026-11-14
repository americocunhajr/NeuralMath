"""Exemplo didático: classificação de frutas por uma rede 2-12-1.
Dependências: numpy e matplotlib. Executar: python codigo_rede_neural.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def relu(s):
    """Ativação da camada oculta: ReLU(s) = max(0,s)."""
    return np.maximum(0, s)


def relu_derivada(s):
    # Em s=0 a ReLU não é derivável; adotamos derivada computacional 0.
    # A convenção define apenas o passo de retropropagação nesse ponto.
    return (s > 0).astype(float)


def sigmoid(s):
    """Ativação da saída, com valores entre 0 e 1."""
    return 1 / (1 + np.exp(-s))


def fronteira_base(m):
    # Curva artificial em cm: nível + ressalto gaussiano + subida logística.
    # As constantes foram escolhidas para criar uma fronteira não linear;
    # não resultam de medidas experimentais nem de ajuste aos dados.
    return (6.68 + 0.74*np.exp(-((m-155)/20)**2)
            + 1.50/(1 + np.exp(-(m-202)/5)))


def construir_dados():
    # 16 massas em cada uma das quatro faixas: a1, a2, l1, l2.
    massas_1 = np.linspace(118, 212, 16)
    # Pequenos deslocamentos determinísticos evitam massas idênticas.
    massas_2 = massas_1 + np.array([
        1.2, -1.1, 0.8, -1.3, 1.0, -0.8, 1.1, -1.0,
        1.3, -0.9, 0.9, -1.2, 1.0, -1.0, 0.8, -0.7
    ])
    k = np.arange(16)

    # Maçãs: duas faixas acima da curva, com oscilações pequenas em cm.
    a1 = np.column_stack([
        massas_1,
        fronteira_base(massas_1) + 0.22 + 0.05*np.sin(0.8*k)
    ])
    a2 = np.column_stack([
        massas_2,
        fronteira_base(massas_2) + 0.58 + 0.06*np.cos(0.45*k)
    ])

    # Laranjas: duas faixas abaixo da curva de referência.
    l1 = np.column_stack([
        massas_1,
        fronteira_base(massas_1) - (0.22 + 0.04*np.cos(0.7*k))
    ])
    l2 = np.column_stack([
        massas_2,
        fronteira_base(massas_2) - (0.55 + 0.05*np.sin(0.5*k))
    ])

    X_treino = np.vstack([a1, a2, l1, l2])
    y_treino = np.array([1.0]*32 + [0.0]*32)

    # Oito exemplos ilustrativos; não constituem teste estatístico amplo.
    X_teste = np.array([
        [121.0, 7.20], [135.0, 7.45],
        [152.0, 7.75], [171.0, 7.32],
        [121.0, 6.28], [152.0, 6.86],
        [189.0, 6.48], [208.0, 7.55]
    ])
    y_teste = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    return X_treino, y_treino, X_teste, y_teste


def prever(X, media, desvio, W1, b1, W2, b2):
    # x_j^* = (x_j - mu_j)/sigma_j, usando estatísticas do treino.
    Xn = (X - media) / desvio
    # Propagação direta em notação matricial (uma linha por fruta).
    h = relu(Xn @ W1.T + b1)
    return sigmoid(h @ W2 + b2)


def treinar(Xn, y_treino, eta=0.05, n_epocas=6000, semente=42):
    rng = np.random.default_rng(semente)
    # Pesos aleatórios pequenos quebram simetria entre os neurônios;
    # vieses iniciais zero. A escala 0.1 é uma escolha didática.
    W1 = 0.1 * rng.standard_normal((12, 2))
    b1 = np.zeros(12)
    W2 = 0.1 * rng.standard_normal(12)
    b2 = 0.0
    historico = []

    for epoca in range(n_epocas):
        # SGD: uma atualização por fruta, em ordem embaralhada a cada época.
        for i in rng.permutation(len(Xn)):
            x, alvo = Xn[i], y_treino[i]

            # Propagação direta: eqs. da camada oculta e da saída.
            z1 = W1 @ x + b1
            h = relu(z1)
            z2 = W2 @ h + b2
            y_hat = sigmoid(z2)

            # Retropropagação: delta2 = dE/dz2 (erro quadrático).
            delta2 = (y_hat - alvo) * y_hat * (1 - y_hat)
            dW2, db2 = delta2 * h, delta2
            # delta1_j = delta2 * W2_j * ReLU'(z1_j).
            delta1 = (delta2 * W2) * relu_derivada(z1)
            dW1 = np.outer(delta1, x)
            db1 = delta1

            # Descida estocástica: theta <- theta - eta*dE/dtheta.
            # Todos os gradientes usam os parâmetros ANTERIORES.
            W1 -= eta * dW1
            b1 -= eta * db1
            W2 -= eta * dW2
            b2 -= eta * db2

        # J: perda média do conjunto ao final de cada época.
        h_todos = relu(Xn @ W1.T + b1)
        y_prev = sigmoid(h_todos @ W2 + b2)
        historico.append(np.mean(0.5 * (y_treino - y_prev)**2))

    return W1, b1, W2, b2, np.array(historico)


def gerar_figura(X_treino, y_treino, X_teste, y_teste,
                  media, desvio, W1, b1, W2, b2, salvar=None):
    # Curva de nível y_hat=0.5: fronteira de decisão APRENDIDA.
    massas = np.linspace(116, 214, 500)
    diametros = np.linspace(6.0, 8.85, 500)
    M, D = np.meshgrid(massas, diametros)
    grade = np.column_stack([M.ravel(), D.ravel()])
    P = prever(grade, media, desvio, W1, b1, W2, b2).reshape(M.shape)

    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.contourf(M, D, P, levels=[0, 0.25, 0.50, 0.75, 1],
                colors=['#f3e0cb', '#ecd5c4', '#dbe7cf', '#cde3bf'],
                alpha=0.95)
    ax.contour(M, D, P, levels=[0.5], colors=['#5a1a8a'],
               linewidths=2.2)

    mask = y_treino == 1.0
    ax.scatter(X_treino[mask, 0], X_treino[mask, 1], marker='o',
               s=48, facecolors='white', edgecolors='black', linewidths=1.1)
    mask = y_treino == 0.0
    ax.scatter(X_treino[mask, 0], X_treino[mask, 1], marker='s',
               s=48, facecolors='white', edgecolors='black', linewidths=1.1)
    mask = y_teste == 1.0
    ax.scatter(X_teste[mask, 0], X_teste[mask, 1], marker='^',
               s=92, color='#1f77b4')
    mask = y_teste == 0.0
    ax.scatter(X_teste[mask, 0], X_teste[mask, 1], marker='x',
               s=110, color='#ff7f0e', linewidths=2.0)

    ax.set_xlabel('Massa (g)')
    ax.set_ylabel('Diâmetro (cm)')
    ax.set_title('Saída da rede neural após o treinamento')
    handles = [
        Line2D([0], [0], marker='o', markersize=9, markerfacecolor='white',
               markeredgecolor='black', linestyle='None',
               label='Maçã - treino'),
        Line2D([0], [0], marker='s', markersize=9, markerfacecolor='white',
               markeredgecolor='black', linestyle='None',
               label='Laranja - treino'),
        Line2D([0], [0], marker='^', markersize=9, color='#1f77b4',
               linestyle='None', label='Maçã - teste'),
        Line2D([0], [0], marker='x', markersize=10, color='#ff7f0e',
               linestyle='None', label='Laranja - teste'),
        Line2D([0], [0], color='#5a1a8a', linewidth=2.2,
               label='Fronteira aprendida')
    ]
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    if salvar:
        fig.savefig(salvar, dpi=200, bbox_inches='tight')
    return fig, ax


if __name__ == '__main__':
    X_treino, y_treino, X_teste, y_teste = construir_dados()
    media, desvio = X_treino.mean(axis=0), X_treino.std(axis=0)
    Xn = (X_treino - media) / desvio
    W1, b1, W2, b2, historico = treinar(Xn, y_treino)
    p_treino = prever(X_treino, media, desvio, W1, b1, W2, b2)
    p_teste = prever(X_teste, media, desvio, W1, b1, W2, b2)
    print('Perda média final:', historico[-1])
    print('Acertos treino:', np.sum((p_treino >= 0.5) == y_treino), '/64')
    print('Acertos teste ilustrativo:',
          np.sum((p_teste >= 0.5) == y_teste), '/8')
    gerar_figura(X_treino, y_treino, X_teste, y_teste,
                 media, desvio, W1, b1, W2, b2)
    plt.show()
