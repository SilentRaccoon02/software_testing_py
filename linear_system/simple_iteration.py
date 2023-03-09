from typing import Tuple

from tools import Matrix, Vector
import tools


def calc_b(a: Matrix, u: float) -> Matrix:
    e = tools.eye(len(a))

    return e - a * u


def stop(norm_b: float, x: Vector, x_new: Vector, eps: float) -> bool:
    return norm_b / (1 - norm_b) * (x_new - x).norm < eps


def simple_iteration(a: Matrix, v_b: Vector, eps: float) -> Tuple:
    n = len(a)

    for i in range(4):
        norm_a = a.norm(i)
        u = 1 / norm_a
        m_b = tools.eye(n) - a * u
        norm_b = m_b.norm(i)

        if norm_b < 1:
            break

    else:
        v_b = a.transpose * v_b
        a = a.transpose * a

        for i in range(4):
            norm_a = a.norm(i)
            u = 1 / norm_a
            m_b = tools.eye(n) - a * u
            norm_b = m_b.norm(i)

            if norm_b < 1:
                break

        else:
            raise Exception('Help...')

    c = v_b * u
    x = Vector(c)
    x_new = m_b * x + c

    k = 0
    while not stop(norm_b, x, x_new, eps):
        k += 1

        x = x_new
        x_new = m_b * x + c

    return x_new, k
