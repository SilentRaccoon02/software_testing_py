from tools import Matrix, Vector


# решить систему
# верхняя диагональ
def top(a: Matrix, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('Incorrect data')

    n = len(a)
    x = Vector([0 for _ in range(n)])

    for i in range(n - 1, -1, -1):
        b_sum = b[i]

        for j in range(i + 1, n):
            b_sum -= a[i, j] * x[j]
        x[i] = b_sum / a[i, i]

    return x


# решить систему
# нижняя диагональ
def bottom(a: Matrix, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('Incorrect data')

    n = len(a)
    x = Vector([0 for _ in range(n)])

    for i in range(n):
        b_sum = b[i]

        for j in range(0, i):
            b_sum -= a[i, j] * x[j]
        x[i] = b_sum / a[i, i]

    return x


def test():
    m_a = Matrix([[1, 2, 4],
                  [0, 3, 5],
                  [0, 0, 6]])

    m_b = Matrix([[1, 0, 0],
                  [2, 4, 0],
                  [3, 5, 6]])

    v_a = Vector([7, 7, 7])

    # top
    # 1.56
    # 0.39
    # 1.17
    print(top(m_a, v_a))
    print()

    # bottom
    # 7
    # -1.75
    # -0.875
    print(bottom(m_b, v_a))


if __name__ == '__main__':
    test()
