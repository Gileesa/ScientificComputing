#
# Wave Equation
#

import numpy as np

L = 1
t = 100
N = 100
dt = 1
c=1

l = L / N

A = np.zeros((100, 100))
    

def initialise_wave(matrix):
    Nx = matrix.shape[1]
    x = np.linspace(0, L, N)
    matrix[0, :] = np.sin(2 * np.pi * x)
    # print(matrix[0])
    return matrix


A = initialise_wave(A)

def propagate_wave(A):
    for j in range(1,t-1):
        print(' j', j)
        print(A[j])
        for i in range(0,N-1):
            print(i)
            print(A[j,i])
            new_u = c**2 * ((dt**2)/(l**2)) * (A[i+1,j] + A[i-1,j] - 2*A[i,j]) - A[i,j-1] + 2*A[i,j]
            A[i,j+1] = new_u
        A[0, j+1] = 0
        A[N-1, j+1] = 0
    print(A)

propagate_wave(A)


