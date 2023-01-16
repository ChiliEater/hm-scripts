import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

############################################
# Aufgabe 5a                               #
############################################

print("Aufgabe 5a:\n")

exact = lambda x: 48 * x **2 + 16 * x - 12
inaccurate = 113
def eps(base, n):
    return 0.5 * base ** (1 - n)

print(f'p(1.46) = {inaccurate}')

print(f'Absoulter Fehler: {np.abs(inaccurate - exact(1.46))}')
print(f'Relativer Fehler: {np.abs(inaccurate - exact(1.46)) / np.abs(exact(1.46))}')
print()

############################################
# Aufgabe 5b                               #
############################################

print("Aufgabe 5b:\n")

exact = lambda x: (48 * x + 16) * x - 12
inaccurate = 114

print(f'p(1.46) = {inaccurate}')

print(f'Absoulter Fehler: {np.abs(inaccurate - exact(1.46))}')
print(f'Relativer Fehler: {np.abs(inaccurate - exact(1.46)) / np.abs(exact(1.46))}')

print(f'EPS: {eps(10, 3)}')

