from .outlier import outlier_check
from .var_analysis import analysis_contribution, analysis_correlation
from datasets import Dataset
import pandas as pd
from .gridsearch import grid_search
from .test import test_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import mae, mape, mse, anova
import numpy as np

import os

def read_data(file_path, col_head):
    # Mendapatkan ekstensi file
    file_ext = os.path.splitext(file_path)[1]

    # Menentukan parameter header
    header = 0 if col_head else None

    # Membaca file
    if file_ext == ".csv":
        df = pd.read_csv(file_path, header=header)
    elif file_ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, header=header)

    return df

def train_model(
        data, 
        check_outlier,
        test_proportion,
        seed,
        y_loc,
        x_loc,
        C,
        gamma,
        epsilon):
    print ("Train model sedang berjalan")
    
    # Ambil data
    df = data
    print ("Data sudah diterima dengan ukuran", df.shape)

    # Cek missing value
    print ("\nMissing value check sedang berjalan")
    missing_values = df[df.isnull().any(axis=1)]
    df = df.dropna()
    print ("Missing values berukuran", missing_values.shape)
    print ("Saat ini data berukuran", df.shape)

    # Cek outlier
    if check_outlier:
        outlier_data, clean_data, Z_outlier = outlier_check(df)
        print ("Saat ini data berukuran", clean_data.shape)
    else:
        clean_data = df
        print ("\nTidak dilakukan pengecekan outlier")

    # Pisahkan data bersih menjadi X dan Y
    Y_clean = clean_data.iloc[:, y_loc]
    print ("\nUkuran y_clean saat ini", Y_clean.shape)
    X_clean = clean_data.iloc[:, x_loc]
    print ("Ukuran x_clean saat ini", X_clean.shape)

    # Analisis kontribusi variabel
    x_cont = analysis_contribution(Y_clean, X_clean)
    print ("\nKontribusi variabel x terhadap y\n", x_cont)

    # Analisis korelasi variabel
    x_corr = analysis_correlation(Y_clean, X_clean)
    print ("\nKorelasi variabel x terhadap y\n", x_corr)

    # Split dataset
    dataset = Dataset.from_pandas(clean_data)
    dataset = dataset.train_test_split(test_size=test_proportion, seed=seed)
    train_data = pd.DataFrame(dataset['train'])
    test_data = pd.DataFrame(dataset['test'])
    Y_train = train_data.iloc[:, y_loc]
    X_train = train_data.iloc[:, x_loc]
    Y_test = test_data.iloc[:, y_loc]
    X_test = test_data.iloc[:, x_loc]
    print ("\nBentuk X_train:", X_train.shape, "dan Y_train:", Y_train.shape)
    print ("Bentuk X_test:", X_test.shape, "dan Y_test:", Y_test.shape)

    # Grid search
    model, params, result_gs = grid_search(X_train, Y_train, X_test, Y_test, C, gamma, epsilon)
    print ("\nModel terbaik adalah", model)

    # Mengambil data support vectors pada model dan lagrange coef serta nilai intercept
    support_vectors = model.support_vectors_
    lagrange_coef = model.dual_coef_
    intercept = model.intercept_

    print ("\nNilai intercept:", intercept[0])

    # Testing model
    yhat = test_model(model, X_test)

    # Plot y vs yhat
    plt.plot(Y_test, '.-b', markersize=5, label="Nilai Aktual (y)")
    plt.plot(yhat, '.-r', markersize=5, label="Nilai Prediksi (yhat)")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Order")
    plt.ylabel("y/yhat")
    plt.savefig("python/img/plot1.png")
    plt.clf()

    # Plot residual
    residual = Y_test - yhat
    plt.plot(residual, '.-b', markersize=5)
    plt.grid(True)
    plt.xlabel("Order")
    plt.ylabel("Residual/Error")
    plt.savefig("python/img/plot2.png")
    plt.clf()

    # Evaluasi model
    MAE = mae(Y_test, yhat)
    MAPE = mape(Y_test, yhat)
    e_MSE = mse(Y_test, yhat)
    
    print ("\nNilai MAE:", MAE)
    print ("Nilai MAPE:", MAPE)
    print ("Nilai MSE:", e_MSE)

    n, k = X_test.shape

    # Hitung anova
    SSR, SSE, SST, dfR, dfE, MSR, MSE, F, R2 = anova(Y_test, yhat, k+1, n)
    print ("\nNilai SSR:", SSR)
    print ("Nilai SSE:", SSE)
    print ("Nilai SST:", SST)
    print ("Nilai dfR:", dfR)
    print ("Nilai dfE:", dfE)
    print ("Nilai MSR:", MSR)
    print ("Nilai MSE:", MSE)
    print ("Nilai F:", F)
    print ("Nilai R2:", R2)

    print ("\n=============== BATAS SUCI ===============")

    print ("\nRegression Equation")
    print("f(x) = sum(lagrange_coef * kernel) + %.8f" % (intercept[0]))

    print ("\nVariables Info")
    print ("Variable\t| Simultaneous\t| Partial")
    print ("Name\t\t| Contribution\t| Correlation")
    print ("---------------------------------------------")
    print ("Intercept\t| %.4f\t|" % (x_cont[0]))
    for i in range(k):
        print ("x%d\t\t| %.4f\t| %.4f" % (i+1, x_cont[i+1], x_corr[i]))

    print ("\nAnalysis of Variance (ANOVA)")
    print ("Source of\t| Sum of\t| Degree of\t| Mean of\t| F-value")
    print ("Variation\t| Squares (SS)\t| freedom (df)\t| Squares (MS)\t|")
    print ("-----------------------------------------------------------------------------")
    print ("Regression\t| %.8f\t| %d\t\t| %.8f\t| %.8f" % (SSR, dfR, MSR, F))
    print ("Error\t\t| %.8f\t| %d\t\t| %.8f\t|" % (SSE, dfE, MSE))
    print ("Total\t\t| %.8f\t| \t\t| \t\t|" % (SST))

    print ("\nEvaluation")
    print ("MAE\t\t| MAPE\t\t| MSE\t\t| R2")
    print ("-------------------------------------------------------------")
    print ("%.8f\t| %.8f\t| %.8f\t| %.8f" % (MAE, MAPE, e_MSE, R2))

    print ("\nModel Info")
    print ("Kernel\t| C\t| Gamma\t| Epsilon")
    print ("----------------------------------")
    print ("RBF\t| %s\t| %s\t| %s" % (params['C'], params['gamma'], params['epsilon']))

    return {
        'raw_data': np.array(data),
        'missing_data': np.array(missing_values),
        'outlier_data': np.array(outlier_data),
        'score_outlier': np.array(Z_outlier),
        'clean_data': np.array(clean_data),
        'train_data': np.hstack((np.array(Y_train).reshape(-1,1), np.array(X_train))),
        'grid_search': np.array(result_gs),
        'support_vectors': np.hstack((support_vectors, lagrange_coef.reshape(-1, 1))),
        'test_results': np.hstack((np.array(Y_test).reshape(-1,1), np.array(X_test), yhat.reshape(-1,1), np.array(residual).reshape(-1,1))),
        'b': intercept[0],
        'x_cont': x_cont,
        'x_corr': x_corr,
        'SSR': SSR,
        'SSE': SSE,
        'SST': SST,
        'dfR': dfR,
        'dfE': dfE,
        'MSR': MSR,
        'MSE': MSE,
        'F': F,
        'MAE': MAE,
        'MAPE': MAPE,
        'e_MSE': e_MSE,
        'R2': R2,
        'kernel': 'RBF',
        'C': params['C'],
        'gamma': params['gamma'],
        'epsilon': params['epsilon']
    }