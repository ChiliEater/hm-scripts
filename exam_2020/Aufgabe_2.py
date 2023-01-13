import numpy as np
import matplotlib.pyplot as plt

############################################
# Aufgabe 2a                               #
############################################

print("Aufgabe 2a:\n")


############################################
# Aufgabe 2b                               #
############################################

print("Aufgabe 2b:\n")


############################################
# Aufgabe 2c                               #
############################################

print("Aufgabe 2c:\n")
# Calculates the condition of a given function
def condition(func, derivative, x):
    return abs(derivative(x)) * abs(x) / abs(func(x))

f = lambda x: x ** 2 * np.sin(x)
fd = lambda x: 2 * x * np.sin(x) + x ** 2 * np.cos(x)
print(condition(f, fd, 0.000000001))

############################################
# Aufgabe 2d                               #
############################################

print("Aufgabe 2d:\n")
x = np.arange(np.pi * -2, np.pi * 3, 0.1)
plt.figure()
plt.plot(x, condition(f, fd, x))
plt.yscale("log")
plt.ylim(0.01, 1000)

############################################
# Aufgabe 2e                               #
############################################

print("Aufgabe 2e:\n")


############################################
# Aufgabe 2f                               #
############################################

print("Aufgabe 2f:\n")

