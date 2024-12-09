import numpy as np

def test_model(model, x_test):
    # Mengubah bentuk masukan menjadi numerik
    x_test = np.array(x_test)

    # Menghitung prediksi
    y_pred = model.predict(x_test)

    return y_pred