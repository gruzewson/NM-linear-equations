import copy
from cmath import sin
import time
import matplotlib.pyplot as plt

index = [1,9,3,5,8,9]
c = index[4]
d = index[5]
e = index[3]
f = index[2]
N = 9 * 100 + c * 10 + d
#a1 = 5 + e
a1 = 3
a2 = a3 = -1


def main():
    #zadanie A
    diag = []
    A = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i][j] = a1
                diag.append(A[i][j])
            elif abs(i - j) == 1:
                A[i][j] = a2
            elif abs(i - j) == 2:
                A[i][j] = a3

    b = [0] * N
    for i in range(N):
        b[i] = sin(i * (f+1))

    start_time = time.time()
    x_jacobi, residuals_jacobi = solve_LU(A, b, diag)
    jacobi_time = time.time() - start_time

    start_time = time.time()
    x_gauss_seidel, residuals_gauss_seidel = solve_gauss_seidel(A, b)
    gauss_seidel_time = time.time() - start_time

    min_length = min(len(residuals_jacobi), len(residuals_gauss_seidel))
    print(f"iterations for Jacobi = {len(residuals_jacobi)}")
    print(f"iterations for Gauss-Seidel = {len(residuals_gauss_seidel)}")

    iterations = range(min_length)
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, residuals_jacobi[:min_length], label='Jacobi')
    plt.semilogy(iterations, residuals_gauss_seidel[:min_length], label='Gauss-Seidel')
    plt.xlabel('Iterations')
    plt.ylabel('Residuum')
    plt.title('change of residuum by iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Time for Jacobi:", jacobi_time)
    print("Time for Gauss-Seidel:", gauss_seidel_time)

#zadanie B
def solve_jacobi(A, b, tolerance=1e-9, max_iterations=100):
        n = len(A)
        x = [1 for _ in range(n)]
        residuals = []

        for _ in range(max_iterations):
            x_new = x.copy()
            for i in range(n):
                s1 = sum(A[i][j] * x[j] for j in range(i))
                s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
                x_new[i] = (b[i] - s1 - s2) / A[i][i]

            x = copy.deepcopy(x_new)

            residual = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
            residual_norm = sum(abs(residual[i]) ** 2 for i in range(n)) ** 0.5
            residuals.append(residual_norm)
            if residual_norm < tolerance:
                break

        return x, residuals


def solve_gauss_seidel(A, b, tolerance=1e-9, max_iterations=1):
    n = len(A)
    x = [1 for _ in range(n)]
    residuals = []

    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        x = copy.deepcopy(x_new)

        residual = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
        residual_norm = sum(abs(residual[i]) ** 2 for i in range(n)) ** 0.5
        residuals.append(residual_norm)
        if residual_norm < tolerance:
            break

    return x, residuals

def solve_LU(A, b, diag, tolerance=1e-9, max_iterations=1):
    U = [row[:] for row in A]  # Make a copy of A for U
    L = [[0] * len(A) for _ in range(len(A))]  # Initialize L as a zero matrix with the same size as A
    n = len(A)
    for i in range(n):
        L[i][i] = 1  # Set the diagonal of L to 1

    for _ in range(max_iterations):
        for i in range(n):
            for j in range(i + 1, n):
                L[j][i] = U[j][i] / U[i][i]  # Compute the entries of L
                for k in range(i, n):
                    U[j][k] -= L[j][i] * U[i][k]  # Update the entries of U

        # Solve Ly = b for y using forward substitution
        y = [0] * n
        for i in range(n):
            y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

        # Solve Ux = y for x using backward substitution
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

        # Calculate the residuals
        residual = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
        residual_norm = sum(abs(residual[i]) ** 2 for i in range(n)) ** 0.5
        if residual_norm < tolerance:
            break

    return x, residual_norm



main()