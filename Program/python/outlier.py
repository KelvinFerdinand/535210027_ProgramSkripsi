import numpy as np

def outlier_check(data):
    print ("\nOutlier check sedang berjalan")

    df = data

    # Deklarasi jumlah baris dan kolom
    num_of_row, num_of_var = df.shape

    # Hitung mean dan stdev
    mean = np.mean(np.array(df), axis=0)
    std = np.std(np.array(df), axis=0)

    # Hitung Z-Score
    Z = (np.array(df) - mean) / std

    # Hitung nilai absolut dari Z-Score
    rawZ = Z
    Z = np.abs(Z)

    # Tetapkan threshold untuk mendeteksi outlier
    threshold = 3

    # Deteksi dan pisahkan outlier
    outlier = np.any(Z > threshold, axis=1)
    data_outliers = df[outlier]
    Z_outliers = rawZ[outlier]
    data_no_outliers = df[~outlier]

    return data_outliers, data_no_outliers, Z_outliers