"""Διδακτικό παράδειγμα: ταξινόμηση φρούτων με δίκτυο 2-12-1.
Απαιτούνται numpy και matplotlib. Εκτέλεση: python neural_network.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def relu(s):
    """Ενεργοποίηση κρυφού στρώματος: ReLU(s) = max(0,s)."""
    return np.maximum(0, s)


def relu_derivative(s):
    # Η ReLU δεν είναι παραγωγίσιμη στο s=0· η υπολογιστική παράγωγος τίθεται 0.
    # Η σύμβαση καθορίζει μόνο το βήμα οπισθοδιάδοσης σε αυτό το σημείο.
    return (s > 0).astype(float)


def sigmoid(s):
    """Ενεργοποίηση εξόδου με τιμές μεταξύ 0 και 1."""
    return 1 / (1 + np.exp(-s))


def reference_curve(m):
    # Τεχνητή καμπύλη (cm): μετατόπιση, γκαουσιανή κορυφή και λογιστική άνοδος.
    # Οι σταθερές επιλέχθηκαν για μια μη γραμμική βοηθητική καμπύλη·
    # δεν είναι μετρήσεις ούτε παράμετροι προσαρμοσμένες σε δεδομένα.
    return (6.68 + 0.74*np.exp(-((m-155)/20)**2)
            + 1.50/(1 + np.exp(-(m-202)/5)))


def construct_data():
    # 16 μάζες σε καθεμία από τις τέσσερις ζώνες: a1, a2, l1, l2.
    masses_1 = np.linspace(118, 212, 16)
    # Μικρές ντετερμινιστικές μετατοπίσεις αποφεύγουν ίδιες μάζες.
    masses_2 = masses_1 + np.array([
        1.2, -1.1, 0.8, -1.3, 1.0, -0.8, 1.1, -1.0,
        1.3, -0.9, 0.9, -1.2, 1.0, -1.0, 0.8, -0.7
    ])
    k = np.arange(16)

    # Μήλα: δύο ζώνες πάνω από την καμπύλη με μικρές αποκλίσεις σε cm.
    a1 = np.column_stack([
        masses_1,
        reference_curve(masses_1) + 0.22 + 0.05*np.sin(0.8*k)
    ])
    a2 = np.column_stack([
        masses_2,
        reference_curve(masses_2) + 0.58 + 0.06*np.cos(0.45*k)
    ])

    # Πορτοκάλια: δύο ζώνες κάτω από τη βοηθητική καμπύλη.
    l1 = np.column_stack([
        masses_1,
        reference_curve(masses_1) - (0.22 + 0.04*np.cos(0.7*k))
    ])
    l2 = np.column_stack([
        masses_2,
        reference_curve(masses_2) - (0.55 + 0.05*np.sin(0.5*k))
    ])

    X_train = np.vstack([a1, a2, l1, l2])
    y_train = np.array([1.0]*32 + [0.0]*32)

    # Οκτώ ενδεικτικά παραδείγματα, όχι πλήρης στατιστική δοκιμή.
    X_test = np.array([
        [121.0, 7.20], [135.0, 7.45],
        [152.0, 7.75], [171.0, 7.32],
        [121.0, 6.28], [152.0, 6.86],
        [189.0, 6.48], [208.0, 7.55]
    ])
    y_test = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    return X_train, y_train, X_test, y_test


def predict(X, mean, std, W1, b1, W2, b2):
    # x_j^* = (x_j - mu_j)/sigma_j, χρησιμοποιώντας στατιστικά μόνο από την εκπαίδευση.
    Xn = (X - mean) / std
    # Εμπρόσθιο πέρασμα σε μητρωική μορφή (μία γραμμή ανά φρούτο).
    h = relu(Xn @ W1.T + b1)
    return sigmoid(h @ W2 + b2)


def train(Xn, y_train, eta=0.05, n_epochs=6000, seed=42):
    rng = np.random.default_rng(seed)
    # Μικρά τυχαία βάρη αίρουν τη συμμετρία των νευρώνων·
    # οι αρχικές πολώσεις είναι μηδενικές. Η κλίμακα 0.1 είναι διδακτική επιλογή.
    W1 = 0.1 * rng.standard_normal((12, 2))
    b1 = np.zeros(12)
    W2 = 0.1 * rng.standard_normal(12)
    b2 = 0.0
    history = []

    for epoch in range(n_epochs):
        # SGD: ενημέρωση ανά φρούτο, αναδιάταξη σε κάθε εποχή.
        for i in rng.permutation(len(Xn)):
            x, target = Xn[i], y_train[i]

            # Εμπρόσθιο πέρασμα: εξισώσεις κρυφού και εξωτερικού στρώματος.
            z1 = W1 @ x + b1
            h = relu(z1)
            z2 = W2 @ h + b2
            y_hat = sigmoid(z2)

            # Οπισθοδιάδοση: delta2 = dE/dz2 (τετραγωνική απώλεια).
            delta2 = (y_hat - target) * y_hat * (1 - y_hat)
            dW2, db2 = delta2 * h, delta2
            # delta1_j = delta2 * W2_j * ReLU'(z1_j).
            delta1 = (delta2 * W2) * relu_derivative(z1)
            dW1 = np.outer(delta1, x)
            db1 = delta1

            # Στοχαστική κατάβαση: theta <- theta - eta*dE/dtheta.
            # Όλες οι κλίσεις χρησιμοποιούν τις ΠΡΟΗΓΟΥΜΕΝΕΣ τιμές παραμέτρων.
            W1 -= eta * dW1
            b1 -= eta * db1
            W2 -= eta * dW2
            b2 -= eta * db2

        # J: μέση απώλεια εκπαίδευσης στο τέλος κάθε εποχής.
        h_all = relu(Xn @ W1.T + b1)
        y_pred = sigmoid(h_all @ W2 + b2)
        history.append(np.mean(0.5 * (y_train - y_pred)**2))

    return W1, b1, W2, b2, np.array(history)


def plot_results(X_train, y_train, X_test, y_test,
                  mean, std, W1, b1, W2, b2, save_path=None):
    # Ισοσταθμική y_hat=0.5: ΜΑΘΗΜΕΝΟ όριο απόφασης.
    masses = np.linspace(116, 214, 500)
    diameters = np.linspace(6.0, 8.85, 500)
    M, D = np.meshgrid(masses, diameters)
    grid = np.column_stack([M.ravel(), D.ravel()])
    P = predict(grid, mean, std, W1, b1, W2, b2).reshape(M.shape)

    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.contourf(M, D, P, levels=[0, 0.25, 0.50, 0.75, 1],
                colors=['#f3e0cb', '#ecd5c4', '#dbe7cf', '#cde3bf'],
                alpha=0.95)
    ax.contour(M, D, P, levels=[0.5], colors=['#5a1a8a'],
               linewidths=2.2)

    mask = y_train == 1.0
    ax.scatter(X_train[mask, 0], X_train[mask, 1], marker='o',
               s=48, facecolors='white', edgecolors='black', linewidths=1.1)
    mask = y_train == 0.0
    ax.scatter(X_train[mask, 0], X_train[mask, 1], marker='s',
               s=48, facecolors='white', edgecolors='black', linewidths=1.1)
    mask = y_test == 1.0
    ax.scatter(X_test[mask, 0], X_test[mask, 1], marker='^',
               s=92, color='#1f77b4')
    mask = y_test == 0.0
    ax.scatter(X_test[mask, 0], X_test[mask, 1], marker='x',
               s=110, color='#ff7f0e', linewidths=2.0)

    ax.set_xlabel('Μάζα (g)')
    ax.set_ylabel('Διάμετρος (cm)')
    ax.set_title('Έξοδος δικτύου μετά την εκπαίδευση')
    handles = [
        Line2D([0], [0], marker='o', markersize=9, markerfacecolor='white',
               markeredgecolor='black', linestyle='None',
               label='Μήλο - εκπαίδευση'),
        Line2D([0], [0], marker='s', markersize=9, markerfacecolor='white',
               markeredgecolor='black', linestyle='None',
               label='Πορτοκάλι - εκπαίδευση'),
        Line2D([0], [0], marker='^', markersize=9, color='#1f77b4',
               linestyle='None', label='Μήλο - δοκιμή'),
        Line2D([0], [0], marker='x', markersize=10, color='#ff7f0e',
               linestyle='None', label='Πορτοκάλι - δοκιμή'),
        Line2D([0], [0], color='#5a1a8a', linewidth=2.2,
               label='Μαθημένο όριο απόφασης')
    ]
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, ax


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = construct_data()
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    Xn = (X_train - mean) / std
    W1, b1, W2, b2, history = train(Xn, y_train)
    p_train = predict(X_train, mean, std, W1, b1, W2, b2)
    p_test = predict(X_test, mean, std, W1, b1, W2, b2)
    print('Τελική μέση απώλεια:', history[-1])
    print('Σωστές προβλέψεις εκπαίδευσης:', np.sum((p_train >= 0.5) == y_train), '/64')
    print('Σωστές ενδεικτικές προβλέψεις δοκιμής:',
          np.sum((p_test >= 0.5) == y_test), '/8')
    plot_results(X_train, y_train, X_test, y_test,
                 mean, std, W1, b1, W2, b2)
    plt.show()
