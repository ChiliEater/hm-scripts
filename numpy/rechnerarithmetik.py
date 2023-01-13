import numpy as np

# Calculates the maximum floating point number based on the given base, number of digits in the mantissa and
# the maximum exponent
#
# Formula: (base ^ e_max) - (base ^ (e_max - n))
def x_max(base: int, n: int, e_max: int) -> float:
    return base ** e_max * (1 - base ** (-n))


# Calculates the smallest positive floating point number based on the given base and the minimum exponent
#
# Formula: base ^ (e_min - 1)
def x_min(base, e_min):
    return base ** (e_min - 1)


# Calculates the largest relative error when rounding a floating point number to a given number of digits
def eps(base, n):
    return base ** (1 - n)


# Calculates the amount of numbers given the length of a mantissa and the length of an eponent (signed)
def all_numbers_signed(mantisse_length, exponent_length, base=2):
    return 2 * base ** (mantisse_length - 1) * (2 * base ** exponent_length - 1) + 1


# Calculates the amount of numbers given the length of a mantissa and the length of an eponent (unsigned)
def all_numbers_unsigned(mantsse_length, exponent_length, base=2):
    return base ** (mantsse_length - 1) * (2 * base ** exponent_length - 1) + 1

# Calculates the condition of a given function
def condition(func, derivative, x):
    return abs(derivative(x)) * abs(x) / abs(func(x))

f = lambda x: x ** 2 * np.sin(x)
fd = lambda x: 2 * x * np.sin(x) + x ** 2 * np.cos(x)
print(condition(f, fd, 0.000000001))
