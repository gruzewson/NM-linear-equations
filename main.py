import copy
from cmath import sin
import time
import matplotlib.pyplot as plt

from solvers import solve_jacobi, solve_gauss_seidel, solve_LU

index = [1,9,3,5,8,9]
c = index[4]
d = index[5]
e = index[3]
f = index[2]


def generate_matrix(N, a1, a2, a3):
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i][j] = a1
            elif abs(i - j) == 1:
                A[i][j] = a2
            elif abs(i - j) == 2:
                A[i][j] = a3

    b = [0] * N
    for i in range(N):
        b[i] = sin(i * (f + 1))

    return A, b


def main():
    #zadanie A
    N = 9 * 100 + c * 10 + d
    a1 = 5 + e
    a2 = a3 = -1
    A,b = generate_matrix(N, a1, a2, a3)

    start_time = time.time()
    x_jacobi, residuals_jacobi = solve_jacobi(A, b)
    jacobi_time = time.time() - start_time

    start_time = time.time()
    x_gauss_seidel, residuals_gauss_seidel = solve_gauss_seidel(A, b)
    gauss_seidel_time = time.time() - start_time

    # zadanie B
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

    # zadanie C i D
    N = 9 * 100 + c * 10 + d
    a1 = 3
    a2 = a3 = -1
    A, b = generate_matrix(N, a1, a2, a3)

    start_time = time.time()
    x_jacobi, residuals_jacobi = solve_jacobi(A, b)
    jacobi_time = time.time() - start_time

    start_time = time.time()
    x_gauss_seidel, residuals_gauss_seidel = solve_gauss_seidel(A, b)
    gauss_seidel_time = time.time() - start_time

    start_time = time.time()
    x_LU, residual_LU = solve_LU(A, b)
    LU_time = time.time() - start_time

    min_length = min(len(residuals_jacobi), len(residuals_gauss_seidel))
    print(f"iterations for Jacobi = {len(residuals_jacobi)}")
    print(f"iterations for Gauss-Seidel = {len(residuals_gauss_seidel)}")
    print(f"residual for LU = {residual_LU}")

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
    print("Time for LU:", LU_time)


main()