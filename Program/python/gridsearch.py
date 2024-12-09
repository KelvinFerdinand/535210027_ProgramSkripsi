import numpy as np
from sklearn.svm import SVR
from .test import test_model
from .evaluate import mae

def grid_search(x_train, y_train, x_test, y_test, C_val, gamma_val, epsilon_val):
    # Mengubah bentuk masukan menjadi numerik
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    # Alokasikan array kosong untuk menampung hasil
    models = []
    results = []

    # Loop untuk mencari model terbaik
    for C in C_val:
        for gamma in gamma_val:
            for epsilon in epsilon_val:
                # Inisialisasi model
                model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)

                # Latih model dengan data latih
                model.fit(x_train, y_train)

                # Uji coba model
                yhat = test_model(model, x_test)

                # Hitung akurasi model
                score = mae(y_test, yhat)

                # Simpan hasil ke dalam list
                models.append(model)
                results.append({
                    'C': C,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'MAE': score
                })

    # Cari model terbaik dari hasil grid search
    best_index = np.argmin([result['MAE'] for result in results])
    best_model = models[best_index]
    best_params = results[best_index]

    return best_model, best_params, results