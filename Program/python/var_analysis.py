import numpy as np

def analysis_contribution(Y,X):
    # Mengubah bentuk data input menjadi array numerik
    Y, X = np.array(Y), np.array(X)

    # Menambahkan 1 kolom di X paling kiri untuk konstanta (intercept)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Hitung koefisien regresi
    # b = (X'X)^-1 X'Y
    b = (np.linalg.inv(X.T @ X)) @ (X.T @ Y)

    return b

def analysis_correlation(Y,X):
    # Mengubah bentuk data input menjadi array numerik
    Y, X = np.array(Y), np.array(X)

    # Melihat jumlah variabel independen
    n, p = X.shape

    # Inisialisasi variabel untuk menyimpan nilai korelasi
    r = []

    # Menghitung korelasi tiap variabel independen ke dependen
    for i in range (p):
        atas = 0
        bawah_kiri = 0
        bawah_kanan = 0

        for j in range (n):
            xj_xbar = X[j,i] - np.mean(X[:,i])
            yj_ybar = Y[j] - np.mean(Y)

            atas += xj_xbar * yj_ybar
            bawah_kiri += xj_xbar ** 2
            bawah_kanan += yj_ybar ** 2
        
        corr = atas / np.sqrt(bawah_kiri * bawah_kanan)
        r.append(corr)

    r = np.array(r)

    return r