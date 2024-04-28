import numpy as np

# Parametry
a1 = 5 + 1  # e = 1
a2 = a3 = -1
c = 1
d = 1
f = 1
N = 9 * c * d

# Macierz A
A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            A[i, j] = a1
        elif abs(i - j) == 1:
            A[i, j] = a2
        elif abs(i - j) == 2:
            A[i, j] = a3

# Wektor b
b = np.array([np.sin((n + 1) * (f + 1)) for n in range(N)])

print("Macierz A:")
print(A)
print("\nWektor b:")
print(b)
