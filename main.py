from cmath import sin

index = [1,9,3,5,8,9]
c = index[4]
d = index[5]
e = index[3]
f = index[2]
N = 9 * 100 + c * 10 + d
a1 = 5 + e
a2 = a3 = -1

A = [[0]*N for _ in range(N)]
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
    b[i] = sin(i * (f+1))

print("Macierz A:")
print(A)
print("\nWektor b:")
print(b)
