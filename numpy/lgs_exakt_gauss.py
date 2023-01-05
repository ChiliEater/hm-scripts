import numpy as np

#######################
# Definitionen        #
#######################

a = np.array([
    [1, 1, 0],
    [3, -1, 2],
    [2, -1, 3],
], dtype=np.float64)

b = np.array([1, 1, 0], dtype=np.float64)

#######################
# LGS lösen           #
#######################

x = np.linalg.solve(a, b)

#######################
# Ausgabe             #
#######################

print(f'x = {x}')
