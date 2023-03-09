from typing import Tuple

from tools import Matrix, Vector
import tools
import solve


def calc_lu(a: Matrix) -> Tuple:
    n = len(a)
    u = Matrix(a)
    l = tools.zeros(n)

    for i in range(n):
        for j in range(i, n):
            l[j, i] = u[j, i] / u[i, i]
    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                l[j, i] = u[j, i] / u[i, i]
        for i in range(k, n):
            for j in range(k - 1, n):
                u[i, j] = u[i, j] - l[i, k - 1] * u[k - 1, j]

    return l, u


def lu(a: Matrix, b: Vector) -> Vector:
    a, b = tools.permutation(a, b)

    l, u = calc_lu(a)
    y = solve.bottom(l, b)
    x = solve.top(u, y)

    return x
