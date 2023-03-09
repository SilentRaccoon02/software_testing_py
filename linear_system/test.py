from typing import List
import pandas as pd
import numpy as np

import tools
from tools import Matrix, Vector
from simple_iteration import simple_iteration
from Seidel import seidel
from LU import lu
from QR import qr

N = 14


def np_solve(a: List[List[float]], b: List[float]) -> Vector:
    np_a = np.array(a)
    np_b = np.array(b)

    return Vector(np.linalg.solve(np_a, np_b).tolist())


def solve(a: List[List[float]], b: List[float], np_result: Vector, test_n: int, extra=(0, 0)) -> List[List[float]]:
    m_a = Matrix(a)
    v_b = Vector(b)
    rows = []

    if extra == (0, 0):
        n_end = 5

    else:
        n_end = 4

    for i in range(2, n_end):
        eps = 10 ** (-i)

        try:
            simple_result, simple_k = simple_iteration(m_a, v_b, eps)

        except OverflowError:
            simple_result = Vector([0 for _ in range(3)])
            simple_k = -1

        seidel_result, seidel_k = seidel(m_a, v_b, eps)
        lu_result = lu(m_a, v_b)
        qr_result = qr(m_a, v_b)

        simple_delta = (simple_result - np_result).abs
        seidel_delta = (seidel_result - np_result).abs
        lu_delta = (lu_result - np_result).abs
        qr_delta = (qr_result - np_result).abs

        if extra == (0, 0):
            if i == 2:
                rows.append([test_n, np_result.str_h, eps, simple_result.str_h, simple_k, seidel_result.str_h, seidel_k,
                             lu_result.str_h, qr_result.str_h])
                rows.append(
                    ['', '', '', simple_delta.str_h, '', seidel_delta.str_h, '', lu_delta.str_h, qr_delta.str_h])

            else:
                rows.append(
                    [test_n, np_result.str_h, eps, simple_result.str_h, simple_k, seidel_result.str_h, seidel_k, '',
                     ''])
                rows.append(['', '', '', simple_delta.str_h, '', seidel_delta.str_h, '', '', ''])

        else:
            if i == 2:
                rows.append([test_n, extra[0], extra[1], np_result.str_h, eps, simple_result.str_h, simple_k,
                             seidel_result.str_h, seidel_k, lu_result.str_h, qr_result.str_h])
                rows.append(['', '', '', '', '', simple_delta.str_h, '', seidel_delta.str_h, '', lu_delta.str_h,
                             qr_delta.str_h])

            else:
                rows.append(
                    [test_n, '', '', np_result.str_h, eps, simple_result.str_h, simple_k, seidel_result.str_h, seidel_k,
                     '', ''])
                rows.append(['', '', '', '', '', simple_delta.str_h, '', seidel_delta.str_h, '', '', ''])

    return rows


def test_5() -> List[List[float]]:
    rows = []

    for n in range(4, 7):
        for e_n in range(2):

            if e_n == 0:
                e = 0.001

            else:
                e = 0.000001

            p_1 = tools.eye(n)
            p_2 = tools.eye(n)

            for i in range(n):
                for j in range(i + 1, n):
                    p_1[i, j] = -1

            for i in range(n):
                for j in range(n):
                    if j > i:
                        p_2[i, j] = -1
                    else:
                        p_2[i, j] = 1

            a = p_1 + p_2 * (e * n)

            b = [float(-1) for _ in range(n)]
            b[n - 1] = float(1)

            rows += solve(a.list, b, np_solve(a.list, b), 5, (n, e))

    return rows


def test(n: int) -> List[List[float]]:
    rows = []

    if n == 0:
        a = [[0, 2, 3],
             [1, 2, 4],
             [4, 5, 6]]
        b = [13, 17, 32]

    elif n == 1:
        a = [[N + 2, 1, 1],
             [1, N + 4, 1],
             [1, 1, N + 6]]
        b = [N + 4, N + 6, N + 8]

    elif n == 2:
        a = [[-(N + 2), 1, 1],
             [1, -(N + 4), 1],
             [1, 1, -(N + 6)]]
        b = [-(N + 4), -(N + 6), -(N + 8)]

    elif n == 3:
        a = [[-(N + 2), N + 3, N + 4],
             [N + 5, -(N + 4), N + 1],
             [N + 4, N + 5, -(N + 6)]]
        b = [-(N + 4), -(N + 6), -(N + 8)]

    elif n == 4:
        a = [[N + 2, N + 1, N + 1],
             [N + 1, N + 4, N + 1],
             [N + 1, N + 1, N + 6]]
        b = [N + 4, N + 6, N + 8]

    else:
        raise ValueError('Invalid data')

    rows += solve(a, b, np_solve(a, b), n)

    return rows


def main():
    columns_1 = ['#test', 'x', 'eps', 'SI_x', 'SI_k', 'Seidel_x', 'Seidel_k', 'LU_x', 'QR_x']
    columns_2 = ['#test', 'n', 'e', 'x', 'eps', 'SI_x', 'SI_k', 'Seidel_x', 'Seidel_k', 'LU_x', 'QR_x']
    rows_2 = test_5()
    rows_1 = []

    for i in range(5):
        rows_1 += test(i)

    table_1 = pd.DataFrame(rows_1, columns=columns_1)
    table_2 = pd.DataFrame(rows_2, columns=columns_2)

    table_1.to_excel('table_1.xlsx')
    table_2.to_excel('table_2.xlsx')

    print(table_1)
    print(table_2)


if __name__ == '__main__':
    main()
