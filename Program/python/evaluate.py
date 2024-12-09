import numpy as np

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / y))

def mse(y, yhat):
    return np.mean(np.square(np.subtract(y, yhat)))

def anova(y, yhat, p, n):
    SST = np.sum((y - np.mean(y)) ** 2)
    SSE = np.sum((yhat - y) ** 2)

    R2 = 1 - (SSE / SST)

    SSR = SST - SSE

    dfR = p-1
    dfE = n-p

    MSR = SSR / dfR
    MSE = SSE / dfE

    F = MSR / MSE

    return SSR, SSE, SST, dfR, dfE, MSR, MSE, F, R2