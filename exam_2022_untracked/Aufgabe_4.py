import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

############################################
# Aufgabe 4a                               #
############################################

print("Aufgabe 4a:\n")
def condition(func, derivative, x):
    return abs(derivative(x)) * abs(x) / abs(func(x))

f = lambda x: x / np.log(x)
fd = lambda x: (np.log(x) - 1) / (np.log(x)**2)

K = condition(f,fd,3)
err = 0.1
x = 3
print(f'K = {K}')
print(f'Fehlerfortpflanzung = {K*(err/x)}')
print()

############################################
# Aufgabe 4b                               #
############################################

print("Aufgabe 4b:\n")

