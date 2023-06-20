import numpy as np
import matplotlib.pyplot as plt

#########################
# Definitionen          #
#########################

a = 2
b = 5
n = 30

y0 = 1
f = lambda x, y: x / y


# Runge-Kutta Klassisch:
#table = np.array([
#    [0,   0,   0,   0,   0  ],
#    [0.5, 0.5, 0,   0,   0  ],
#    [0.5, 0,   0.5, 0,   0  ],
#    [1,   0,   0,   1,   0  ],
#    [0,   1/6, 1/3, 1/3, 1/6]
#])

# c_1 | a_11 a_12 ... a_1s
# c_2 | a_21 a_22 ... a_2s
# ... | ...
# c_s | a_s1 a_s2 ... a_ss
# ----|--------------------
#     | b_1  b_2  ... b_s
table = np.array([
    [0,   0,   0,   0],
    [1/3, 1/3, 0,   0],
    [2/3, 0,   2/3, 0],
    [0,   1/4, 0,   3/4]
])

s = table.shape[0] - 1
c_a = table[:-1, 1:]
c_b = table[-1, 1:]
c_c = table[:-1, 0]

#########################
# Runge-Kutta Verfahren #
#########################
h_to_n = lambda n, a, b: np.rint((b - a) / n)

def runge_kutta(a, b, n, y0, f):
    step_size = (b - a) / n

    x = np.array([a])
    y = np.array([y0])

    for i in range(int(n)):
        k = np.zeros(s)
        k[0] = f(x[-1], y[-1])

        for j in range(1, s):
            x_j = x[-1] + c_c[j] * step_size
            y_j = y[-1] + step_size * np.sum(c_a[j, :j] * k[:j])
            k[j] = f(x_j, y_j)

        x = np.append(x, x[-1] + step_size)
        y = np.append(y, y[-1] + step_size * np.sum(c_b * k))
    return x, y

x, y = runge_kutta(a, b, n, y0, f)

y_real = lambda x: np.sqrt(x**2 - 3.0)
err = lambda yb, yr: np.abs(yb - yr)
print("X:")
print(x)
print("Y:")
print(y)
print("Error:")
print(np.abs(y[-1] - y_real(b)))

plt.figure()
plt.plot(x,y, label="Runge")
plt.plot(x, y_real(x), label='Real')
plt.legend()
plt.grid()
plt.show()

#sizes = np.array([1, 0.1, 0.01, 0.001])
#sizes = h_to_n(sizes, a, b)
#yb = np.array([])
#for ni in sizes:
#    _, y = runge_kutta(a,b,ni,y0,f)
#    yb = np.append(yb, y[-1])
#errors = err(yb, y_real(b))
#plt.figure()
#plt.plot(sizes, errors)
#plt.show()