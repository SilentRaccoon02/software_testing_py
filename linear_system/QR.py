from typing import Tuple

from tools import Matrix, Vector
import tools
import solve


def calc_qr(a: Matrix) -> Tuple:
    n = len(a)
    r = Matrix(a)
    q = tools.eye(n)

    q_all = [q]

    for i in range(n - 1):
        y = tools.column(r, i)
        y.cut(i)
        alpha = y.norm
        z = tools.ort(n - i, 0)

        rho = (y - z * alpha).norm
        w = (y - z * alpha) / rho

        e = tools.eye(n - i)
        q = e - (w * w.transpose) * float(2)

        r_temp = Matrix(r)
        r_temp.cut(i)
        r_temp = q * r_temp

        for j in range(n - i):
            for k in range(n - i):
                r[i + j, i + k] = r_temp[j, k]

        q = tools.extend(q, n)
        q_all.append(q)

    q = Matrix(q_all[0])

    for i in range(1, len(q_all)):
        q *= q_all[i]

    return q, r


def qr(a: Matrix, b: Vector) -> Vector:
    q, r = calc_qr(a)
    y = q.transpose * b
    x = solve.top(r, y)

    return x
