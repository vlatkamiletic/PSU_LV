import numpy as np
import matplotlib.pyplot as plt

# Definiranje koordinata točaka A, B, C i D
A = np.array([1, 1])
B = np.array([3, 1])
C = np.array([3, 2])
D = np.array([2, 2])

# Stvaranje x i y koordinata za crtanje linija
x_coords = [A[0], B[0], C[0], D[0], A[0]]  # dodajemo A[0] na kraju kako bismo zatvorili oblik
y_coords = [A[1], B[1], C[1], D[1], A[1]]

# Crtanje četverokuta s posebnim opcijama za liniju
plt.plot(x_coords, y_coords, 'b', linewidth=1.5, marker=".", markersize=10)

# Postavljanje granica osi i prikaz grafa
plt.axis([0, 4, 0, 4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()
