import numpy as np
import matplotlib.pyplot as plt
import IPython.display as dp
import sympy as sp

############################################
# Aufgabe 1a                               #
############################################

print("Aufgabe 1a:\n")

alpha = 2.3
a = np.array([
    [1, alpha, 0, -alpha, ],
    [alpha, 1, -alpha, 0, ],
    [0, -alpha, 1, alpha, ],
    [-alpha, 0, alpha, 1, ],
], dtype=np.float64)

b = np.array([
    [2.3, ],
    [-1.3, ],
    [2.7, ],
    [-1.7, ],
], dtype=np.float64)

x0 = np.array([
    [-1, ],
    [-1, ],
    [1, ],
    [-1, ],
], dtype=np.float64)

n = 1

# Solve

s = a.shape

x = x0
for _ in range(n):
    x_new = np.zeros(s[0])
    for i in range(s[0]):
        x_new[i] = (1 / a[i, i]) * (b[i] - np.sum(a[i, :i] * x[:i]) - np.sum(a[i, i + 1:] * x[i + 1:]))
    x = x_new

# Output

print(f'x = {x}')


############################################
# Aufgabe 1b                               #
############################################

print("Aufgabe 1b:\n")

def err(func, x, error):
    return func(x + error) * func(x - error)

############################################
# Aufgabe 1c                               #
############################################

print("Aufgabe 1c:\n")


############################################
# Aufgabe 1d                               #
############################################

print("Aufgabe 1d:\n")


############################################
# Aufgabe 1e                               #
############################################

print("Aufgabe 1e:\n")


############################################
# Aufgabe 1f                               #
############################################

print("Aufgabe 1f:\n")

