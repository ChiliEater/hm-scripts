import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import IPython.display as dp


def backwards_substitution(u: sp.Matrix, b: sp.Matrix, symbol: str = 'x', output: bool = False) -> sp.Matrix:
    n = len(b)
    x = sp.zeros(n, 1)

    for i in range(n - 1, -1, -1):
        zero_row = all([x.is_zero for x in u[i, :]])
        if zero_row:
            x[i] = sp.Symbol(f'{symbol}{i + 1}')

            if output:
                dp.display(dp.Math(f'{symbol}_{{{i + 1}}} = {sp.latex(x[i])}'))

            continue

        s = sp.Number(0)
        if i != n - 1:
            s = u[i, i + 1:n].dot(x[i + 1:n])

        x[i] = (b[i] - s) / u[i, i]

        if output:
            eq = f'{symbol}_{{{i + 1}}} = '
            if not s.is_zero:
                eq += f'\\frac{{{sp.latex(b[i])} - ({sp.latex(s)})}}{{{sp.latex(u[i, i])}}} = '
            eq += f'\\frac{{{sp.latex(b[i] - s)}}}{{{sp.latex(u[i, i])}}} = {sp.latex(x[i])}'

            dp.display(dp.Math(eq))

    return x


def forwards_substitution(l: sp.Matrix, b: sp.Matrix, symbol: str = 'x', output: bool = False) -> sp.Matrix:
    n = len(b)
    x = sp.zeros(n, 1)

    for i in range(n):
        zero_row = all([x.is_zero for x in l[i, :]])
        if zero_row:
            x[i] = sp.Symbol(f'{symbol}{i + 1}')

            if output:
                dp.display(dp.Math(f'{symbol}_{{{i + 1}}} = {sp.latex(x[i])}'))

            continue

        s = sp.Number(0)
        if i != 0:
            s = l[i, 0:i].dot(x[0:i])

        x[i] = (b[i] - s) / l[i, i]

        if output:
            eq = f'{symbol}_{{{i + 1}}} = '
            if not s.is_zero:
                eq += f'\\frac{{{sp.latex(b[i])} - ({sp.latex(s)})}}{{{sp.latex(l[i, i])}}} = '
            eq += f'\\frac{{{sp.latex(b[i] - s)}}}{{{sp.latex(l[i, i])}}} = {sp.latex(x[i])}'

            dp.display(dp.Math(eq))

    return x


def swap_row(mat: sp.Matrix, i: int, j: int):
    mat[i, :], mat[j, :] = mat[j, :], mat[i, :]


def find_pivot(col: sp.Matrix) -> int:
    max_idx = 0
    for i in range(1, len(col)):
        if col[max_idx].is_zero or (len(col[i].free_symbols) == 0 and col[i] > col[max_idx]):
            max_idx = i

    return max_idx


def pivot(mat: sp.Matrix, perm: dict[str, sp.Matrix], j: int, output: bool = False) -> int:
    col = abs(mat[j:, j])
    max_idx = find_pivot(col)

    if max_idx != 0:
        perm_str = ' \\; '.join([f'{n} = {sp.latex(p)}' for n, p in perm.items()])
        prev_matrix = sp.latex(mat) + ' \\; ' + perm_str

        swap_row(mat, j, j + max_idx)
        for p in perm.values():
            swap_row(p, j, j + max_idx)

        if output:
            perm_str = ' \\; '.join([f'{n} = {sp.latex(p)}' for n, p in perm.items()])

            dp.display(dp.Math(
                f'\\text{{pivot: swapping rows {j + 1} and {j + max_idx + 1} in }}{prev_matrix} \\text{{ gives }}' +
                sp.latex(mat) + ' \\; ' + perm_str))




############################################
# Aufgabe 3a                               #
############################################

print("Aufgabe 3a:\n")
alpha = sp.var("alpha")
A = sp.Matrix([
    [4, -alpha, ],
    [-3, alpha, ],
])

b = sp.Matrix([
    [1, ],
    [-2, ],
])


A_inv = 1 / (4*alpha - ((-3) * (-alpha))) * sp.Matrix([[alpha, alpha],[3,4]])

def cond(matrix):
    return matrix.norm(ord=1) * matrix.inv().norm(ord=1)

print(cond(A.replace(alpha, 5)))

############################################
# Aufgabe 3b                               #
############################################

print("Aufgabe 3b:\n")

def sign(x):
    if x >= 0:
        return 1
    return -1

def gen_householder_matrix(a: sp.Matrix, i: int, precision: int = -1, output: bool = False) -> sp.Matrix:
    ai = a[:, 0]
    ei = sp.eye(ai.rows)[:, 0]

    ai_norm = sp.sqrt(sum(ai[i] ** 2 for i in range(len(ai))))


    v = ai + sign(ai[0]) * ai_norm * ei
    u = 1 / v.norm() * v

    if precision == -1:
        u = sp.simplify(u)
    else:
        u = u.evalf(precision)

    h = sp.eye(ai.rows) - 2 * (u @ sp.transpose(u))
    if precision == -1:
        h = sp.simplify(h)

    if output:
        v_str = f'v_{i + 1} = {sp.latex(v)}, \\; |v_{i + 1}| = {sp.latex(v.norm())} '
        if precision == -1:
            v_str += '= ' + sp.latex(sp.simplify(v.norm()))
        else:
            v_str += '\\approx ' + sp.latex(v.norm().evalf(precision))

        dp.display(dp.Math(f'A_{i + 1} = {sp.latex(a)} \\; \\rightarrow \\; a_{i + 1} = {sp.latex(ai)}'))
        dp.display(dp.Math(v_str))
        dp.display(dp.Math(f'u_{i + 1} = {sp.latex(u)} \\; \\rightarrow \\; u_{i + 1}^T = {sp.latex(sp.transpose(u))}'))
        dp.display(dp.Math(
            f'H_{i + 1} = I_{len(ai)} - 2 u_{i + 1} u_{i + 1}^T = I_{len(ai)} - 2 \\cdot {sp.latex(u @ sp.transpose(u))} '
            f'= {sp.latex(h)}'))

    return h


def expand_matrix(mat: sp.Matrix, n: int):
    offset = n - mat.rows

    res = sp.eye(n)
    res[offset:, offset:] = mat
    return res


def qr_decompose(a: sp.Matrix, precision: int = -1, output: bool = False) -> (sp.Matrix, sp.Matrix):
    n = a.rows

    r = a.copy()
    q = sp.eye(n)

    for i in range(n - 1):
        if output:
            dp.display(dp.Markdown(f'Iteration {i + 1}'))

        hi = gen_householder_matrix(r[i:, i:], i, precision, output)
        qi = expand_matrix(hi, n)

        r = qi @ r
        q = q @ qi

        if precision == -1:
            r = sp.simplify(r)
            q = sp.simplify(q)
        else:
            # fill lower column with zeros to fix floating point errors
            r = sp.Matrix(r)
            r[(i + 1):, i] = sp.zeros(n - i - 1, 1)

        if output:
            dp.display(dp.Math(
                f'Q_{i + 1} = {sp.latex(qi)} \\rightarrow Q = Q \\cdot Q_{i + 1}^T = {sp.latex(q)}, \\; R = Q_{i + 1} '
                f'\\cdot R = {sp.latex(r)}'))

    return q, r


def qr(a: sp.Matrix, b: sp.Matrix, precision: int = -1, output: bool = False) -> [sp.Matrix, sp.Matrix, sp.Matrix]:
    if output:
        dp.display(dp.Math(f'A = {sp.latex(a)}, \\quad b = {sp.latex(b)}'))
        dp.display(dp.Markdown('## QR-Zerlegung'))

    q, r = qr_decompose(a, precision, output=output)

    if output:
        dp.display(dp.Markdown('Resultat'))
        dp.display(dp.Math(f'Q = {sp.latex(q)}, \\quad R = {sp.latex(r)}'))

    y = sp.transpose(q) @ b

    if output:
        dp.display(dp.Markdown('## Rückwärtseinsetzen'))

    x = backwards_substitution(r, y, output=output)

    if output:
        dp.display(dp.Math('x = ' + sp.latex(x)))

    return q, r, sp.simplify(x)





alpha = sp.var("alpha")
a = sp.Matrix([
    [4, -alpha, ],
    [-3, alpha, ],
])

b = sp.Matrix([
    [1, ],
    [-2, ],
])
b = sp.Matrix([9, -4, sp.var('z')])

q, r, x = qr(a, b, precision=5, output=True)