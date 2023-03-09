from typing import Tuple

from tools import Matrix, Vector
import tools


def calc_x_new(c: Matrix, d: Vector, x: Vector) -> Vector:
    n = len(x)
    x_new = Vector([0 for _ in range(n)])

    for i in range(n):
        temp = 0
        temp_new = 0

        for j in range(i):
            temp += c[i, j] * x_new[j]

        for k in range(i + 1, n):
            temp_new += c[i, k] * x[k]

        x_new[i] = temp + temp_new + d[i]

    return x_new


def stop(a: Matrix, b: Vector, x: Vector, eps: float) -> bool:
    return (a * x - b).norm < eps


def seidel(a: Matrix, b: Vector, eps: float) -> Tuple:
    b = a.transpose * b
    a = a.transpose * a

    n = len(a)
    c = tools.zeros(n)
    d = Vector([0 for _ in range(n)])

    for i in range(n):
        for j in range(n):
            if i != j:
                c[i, j] = - (a[i, j] / a[i, i])

    for i in range(n):
        d[i] = b[i] / a[i, i]

    x = Vector(d)
    x_new = calc_x_new(c, d, x)

    k = 0
    while not stop(a, b, x, eps):
        k += 1

        x = Vector(x_new)
        x_new = calc_x_new(c, d, x)

    return x_new, k
