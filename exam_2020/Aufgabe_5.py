import numpy as np
import matplotlib.pyplot as plt

############################################
# Aufgabe 5a                               #
############################################

print("Aufgabe 5a:\n")

n = 6
c = 4.0

def set_diagonal(num, off_x: int, off_y: int, matrix: np.ndarray):
    x = 0 + off_x
    y = 0 + off_y
    try:
        while True:
            matrix[y][x] = num
            x += 1
            y += 1
    except:
        pass
    return matrix

a = np.eye(n,n)
a = set_diagonal(-1, 0,-1,a)
a = set_diagonal(-1, 0,1,a)
a = set_diagonal(-1, 1,0,a)
a = set_diagonal(c, 0,0,a)
print(a)


############################################
# Aufgabe 5b                               #
############################################

print("Aufgabe 5b:\n")


############################################
# Aufgabe 5c                               #
############################################

print("Aufgabe 5c:\n")


############################################
# Aufgabe 5d                               #
############################################

print("Aufgabe 5d:\n")


############################################
# Aufgabe 5e                               #
############################################

print("Aufgabe 5e:\n")


############################################
# Aufgabe 5f                               #
############################################

print("Aufgabe 5f:\n")

