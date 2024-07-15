import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fungsi untuk melakukan Gaussian elimination pada koefisien.
def gaussian_elimination(A, B):
    if A.shape[0] != A.shape[1]:
        print("Matrix A is not square.")
        return None
    if A.shape[0] != B.shape[0]:
        print("Matrix A and B are not compatible.")
        return None
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]
            B[j] -= factor * B[i]
    return A, B

# Fungsi untuk melalukan back subs untuk menyelesaikan sistem linier.
def back_substitution(A, B):
    n = len(B)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = B[i]
        for j in range(i+1, n):
            X[i] -= A[i, j] * X[j]
        X[i] /= A[i, i]
    return X

#Load and read the data
data = pd.read_csv('aol_data.csv')
months = np.arange(1, 145)
production = data.values.flatten()

# print(data)

# Fungsi untuk melakukan regresi polinomial dengan derajat tertentu.
def polynomial_regression(degree):
    n = len(months)
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)
    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(months ** (i + j))
        B[i] = np.sum(production * months ** i)
    A, B = gaussian_elimination(A, B)
    coefficients = back_substitution(A, B)
    return coefficients

# Fungsi untuk mengkalkulasi mean squared error
def mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Memprediksi produksi dengan menggunakan model polinomial
degree = 4
coefficients = polynomial_regression(degree)
predicted_production = sum(coefficients[i] * months ** i for i in range(degree + 1))
mse = mean_squared_error(production, predicted_production)
print(f"Degree: {degree}, Mean Squared Error: {mse:.2f}")

# Plot data original dan data hasil polinomial regresi
plt.plot(months, production, 'o', label='Actual Data', markersize=3)
plt.plot(months, predicted_production, '-', label='Polynomial Regression')
plt.legend()
plt.show()

# Fungsi untuk melakukan bisection method untuk mencari root
def bisection_method(func, x_low, x_high, tol=0.01, max_iter=100):
    for _ in range(max_iter):
        x_mid = (x_low + x_high) / 2
        if func(x_mid) * func(x_low) < 0:
            x_high = x_mid
        else:
            x_low = x_mid
        if abs(func(x_mid)) < tol:
            break
    return x_mid

# Fungsi untuk mencari bulan ke berapa produksi melebihi 25,000
def find_threshold_month(threshold):
    def production_func(month):
        return sum(coefficients[i] * month ** i for i in range(degree + 1)) - threshold
    return bisection_method(production_func, 1, 250)

threshold_month = find_threshold_month(25000)
start_build_month = threshold_month - 13
print(f"Production will reach 25,000 at month: {threshold_month:.0f}")
print(f"Warehouse should be built starting from month: {start_build_month:.0f}")

# Plot data produksi yang diextend termasuk prediksinya
extended_months = np.arange(1, int(threshold_month) + 1)
extended_production = np.concatenate([production, [sum(coefficients[i] * m ** i for i in range(degree + 1)) for m in range(145, int(threshold_month) + 1)]])
predicted_production_extended = sum(coefficients[i] * extended_months ** i for i in range(degree + 1))

plt.plot(extended_months, extended_production, 'o', label='Extended Data', markersize=3)
plt.plot(extended_months, predicted_production_extended, '-', label='Polynomial Regression')
plt.title("Extended Production Data")
plt.legend()
plt.show()
