import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

############################################
# Aufgabe 6a                               #
############################################

print("Aufgabe 6a:\n")

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

f = lambda x: (np.exp(x) + np.exp(-x)) / 2 - x - 2
df = lambda x: np.exp(x) / 2 - 1 - np.exp(-x) / 2
tol = 10 ** (-7)
stop = lambda f, x, tol: f(x - tol) * f(x + tol) <= 0

x = np.arange(92,200,.1)
plt.figure()
plt.plot(x, f(x))
plt.grid()
plt.show()

x0 = 1
x1 = 1.1
n = 17

# Sekanten-Verfahren                       

x_prev = x0
x_curr = x1

for i in range(n):
    x = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
    x_prev = x_curr
    x_curr = x

    print(f'x_{i + 1} = {x}')

print(f'f(2.0851860141732455) = {f(2.0851860141732455)}')
print()

############################################
# Aufgabe 6b                               #
############################################

print("Aufgabe 6b:\n")
f = lambda x: (np.exp(x) + np.exp(-x)) / 2 - x - 2
df = lambda x: np.exp(x) / 2 - 1 - np.exp(-x) / 2

x0 = -1
n = 5

# Newton-Verfahren                         

x = x0
for i in range(n):
    x -= f(x) / df(x)
    print(f'x_{i + 1} = {x}')

print()
print(f'f(-0.7252637249208065) = {f(-0.7252637249208065)}')