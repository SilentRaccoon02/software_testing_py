from typing import List, Tuple, overload
from copy import deepcopy
from math import sqrt

# количество знаков в строковых представлениях
SIGNS = 7


# класс работает с векторами
class Vector:
    # инициализация с помощью массива
    @overload
    def __init__(self, data: List[float]):
        ...

    # инициализация с помощью вектора
    @overload
    def __init__(self, data: 'Vector'):
        ...

    def __init__(self, data: 'List[float] | Vector'):
        if isinstance(data, List):
            if not len(data) > 0:
                raise ValueError('Data is empty')

            self.__size = len(data)
            self.__vector = deepcopy(data)

        elif isinstance(data, Vector):
            self.__size = data.__size
            self.__vector = deepcopy(data.__vector)

        else:
            raise TypeError('Invalid type')

    # длина
    def __len__(self) -> int:
        return self.__size

    # доступ к элементу по индексу
    def __getitem__(self, item: int) -> float:
        if item < 0 or item >= len(self):
            raise IndexError('Vector index out of range')

        return self.__vector[item]

    # изменение элемента по индексу
    def __setitem__(self, key: int, value: float):
        if key < 0 or key >= len(self):
            raise IndexError('Vector index out of range')

        self.__vector[key] = value

    # строковое представление
    def __str__(self) -> str:
        result = '['

        for i in range(len(self)):
            if i > 0:
                result += ' '

            result += f'[{self[i]:.{SIGNS}f}]\n'

        return result[:-1] + ']'

    # умножение на число
    @overload
    def __mul__(self, other: float) -> 'Vector':
        ...

    # умножение на строку
    @overload
    def __mul__(self, other: 'Matrix') -> 'Matrix':
        ...

    def __mul__(self, other: 'float | Matrix'):
        n = len(self)

        if isinstance(other, float):
            result = Vector([0 for _ in range(n)])

            for i in range(n):
                result[i] = self[i] * other

            return result

        if isinstance(other, Matrix):
            if other.square:
                raise ValueError('Matrix is not a row')

            if len(self) != len(other):
                raise ValueError('Vector multiplication is impossible')

            result = zeros(n)

            for i in range(n):
                for j in range(n):
                    result[i, j] = self[i] * other[0, j]

            return result

        raise TypeError('Invalid type')

    # деление на число
    def __truediv__(self, other: float) -> 'Vector':
        # проверки выполнятся в умножении
        return self * (1 / other)

    # сложение
    def __add__(self, other: 'Vector') -> 'Vector':
        if len(self) != len(other):
            raise ValueError('Addition is impossible')

        n = len(self)
        result = Vector([0 for _ in range(n)])

        for i in range(n):
            result[i] = self[i] + other[i]

        return result

    # вычитание
    def __sub__(self, other: 'Vector') -> 'Vector':
        # проверки выполнятся в сложении
        return self + other * float(-1)

    # горизонтальная строка
    @property
    def str_h(self) -> str:
        result = '['

        for i in range(len(self)):
            result += f'{self[i]:.{SIGNS}f} '

        return result[:-1] + ']'

    # вектор модулей
    @property
    def abs(self) -> 'Vector':
        n = len(self)
        result = Vector([0 for _ in range(n)])

        for i in range(n):
            result[i] = abs(self[i])

        return result

    # транспонирование
    @property
    def transpose(self) -> 'Matrix':
        n = len(self)
        result = Matrix([[0 for _ in range(n)]])

        for i in range(n):
            result[0, i] = self[i]

        return result

    # евклидова норма
    @property
    def norm(self) -> float:
        result = 0

        for i in range(len(self)):
            result += self[i] ** 2

        return sqrt(result)

    # обрезать n начальных элементов
    def cut(self, n: int):
        if n < 0 or len(self) <= n:
            raise ValueError('Invalid size')

        self.__size -= n
        self.__vector = deepcopy(self.__vector[n:])


# создать орт
def ort(size: int, i: int) -> Vector:
    if not size > 0 or size <= i or i < 0:
        raise ValueError('Invalid size')

    result = Vector([0 for _ in range(size)])
    result[i] = 1

    return result


# класс работает с квадратными матрицами и строками
# не все операции поддерживаются для строк
class Matrix:
    # инициализация с помощью двумерного массива
    @overload
    def __init__(self, data: List[List[float]]):
        ...

    # иницализация с помощью матрицы
    @overload
    def __init__(self, data: 'Matrix'):
        ...

    def __init__(self, data: 'List[List[float]] | Matrix'):
        if isinstance(data, List):

            if not len(data) > 0:
                raise ValueError('Data is empty')

            for item in data:
                if not len(item) > 0 or len(item) != len(data[0]):
                    raise ValueError('Data is not a matrix')

            if len(data) == 1:
                self.__square = False

            elif len(data) == len(data[0]):
                self.__square = True

            else:
                raise ValueError('Matrix is not a square or a row')

            self.__size = len(data[0])
            self.__matrix = deepcopy(data)

        elif isinstance(data, Matrix):
            self.__square = data.__square
            self.__size = data.__size
            self.__matrix = deepcopy(data.__matrix)

        else:
            raise TypeError('Invalid type')

    # размерность
    def __len__(self) -> int:
        return self.__size

    # является квадратной
    @property
    def square(self) -> bool:
        return self.__square

    # доступ к элементу по индексу
    def __getitem__(self, item: Tuple) -> float:
        self.__check_index(item)

        return self.__matrix[item[0]][item[1]]

    # изменение элемента по индексу
    def __setitem__(self, key: Tuple, value: float):
        self.__check_index(key)
        self.__matrix[key[0]][key[1]] = value

    # строковое представление
    def __str__(self) -> str:
        columns = len(self)

        if self.square:
            rows = len(self)

        else:
            rows = 1

        result = '['

        for i in range(rows):
            if i > 0:
                result += ' '

            result += '['

            for j in range(columns):
                result += f'{self[i, j]:.{SIGNS}f} '

            result = result[:-1] + ']\n'

        return result[:-1] + ']'

    # умножение квадратной матрицы на число
    @overload
    def __mul__(self, other: float) -> 'Matrix':
        ...

    # умножение квадратной матрицы на матрицу
    @overload
    def __mul__(self, other: 'Matrix') -> 'Matrix':
        ...

    # умножение на вектор
    @overload
    def __mul__(self, other: 'Vector'):
        ...

    def __mul__(self, other: 'float | Matrix | Vector'):
        n = len(self)

        if isinstance(other, float):
            if not self.square:
                raise ValueError('Row to value multiplication in not supported')

            result = zeros(n)

            for i in range(n):
                for j in range(n):
                    result[i, j] = self[i, j] * other

            return result

        if isinstance(other, Matrix):
            if not self.square:
                raise ValueError('Row to matrix multiplication in not supported')

            if len(self) != len(other):
                raise ValueError('Matrix multiplication is impossible')

            result = zeros(n)

            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i, j] += self[i, k] * other[k, j]

            return result

        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError('Matrix multiplication is impossible')

            if self.square:
                result = Vector([0 for _ in range(n)])
                for i in range(n):
                    for j in range(n):
                        result[i] += self[i, j] * other[j]

            else:
                result = 0
                for i in range(n):
                    result += self[0, i] * other[i]

            return result

        raise TypeError('Invalid type')

    # деление квадратной матрицы на число
    def __truediv__(self, value: float) -> 'Matrix':
        # проверки выполнятся в умножении
        return self * (1 / value)

    # сложение квадратных матриц
    def __add__(self, other: 'Matrix') -> 'Matrix':
        if not self.square or not other.square:
            raise ValueError('Row addition is not supported')

        if len(self) != len(other):
            raise ValueError('Addition is impossible')

        n = len(self)
        result = zeros(n)

        for i in range(n):
            for j in range(n):
                result[i, j] = self[i, j] + other[i, j]

        return result

    # вычитание квадратных матриц
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        # проверки выполнятся в сложении
        return self + other * float(-1)

    # двумерный массив
    @property
    def list(self) -> List[List[float]]:
        return self.__matrix

    # транспонирование
    @property
    def transpose(self):
        n = len(self)

        if self.square:
            result = zeros(n)

            for i in range(n):
                for j in range(n):
                    result[j, i] = self[i, j]

            return result

        else:
            result = Vector([0 for _ in range(n)])

            for i in range(n):
                result[i] = self[0, i]

            return result

    # перестановка
    @property
    def permutation(self) -> 'Matrix':
        if not self.square:
            raise ValueError('Permutation is impossible')

        n = len(self)
        result = eye(n)
        temp = Matrix(self)

        for i in range(n):
            if self[i, i] == 0:
                for j in range(i, n):
                    if self[i, j] != 0:
                        result.__matrix[i], result.__matrix[j] = result.__matrix[j], result.__matrix[i]
                        temp.__matrix[i], temp.__matrix[j] = temp.__matrix[j], temp.__matrix[i]

                        break

        return result

    def __check_index(self, key: Tuple):
        if len(key) != 2:
            raise IndexError('Invalid index')

        if key[0] < 0 or key[0] >= len(self) or key[1] < 0 or key[1] >= len(self):
            raise IndexError('Matrix index out of range')

    # норма
    def norm(self, i: int) -> float:
        n = len(self)
        result = 0

        if i == 0:
            for i in range(n):
                for j in range(n):
                    if abs(self[i, j]) > result:
                        result = abs(self[i, j])

            return result

        if i == 1:
            for i in range(n):
                temp = 0

                for j in range(n):
                    temp += abs(self[i, j])

                if temp > result:
                    result = temp

            return result

        if i == 2:
            a = self.transpose

            for i in range(n):
                temp = 0

                for j in range(n):
                    temp += abs(a[i, j])

                if temp > result:
                    result = temp

            return result

        if i == 3:
            for i in range(n):
                for j in range(n):
                    result += abs(self[i, j])

            return sqrt(result)

        raise ValueError('Invalid data')

    # обрезать n начальных элементов
    def cut(self, size: int):
        if size < 0 or len(self) <= size:
            raise ValueError('Invalid size')

        if not self.square:
            raise ValueError('Row cut in not supported')

        self.__size -= size
        self.__matrix = deepcopy(self.__matrix[size:])

        for i in range(len(self)):
            self.__matrix[i] = deepcopy(self.__matrix[i][size:])


# создать нулевую матрицу
def zeros(size: int) -> Matrix:
    if not size > 0:
        raise ValueError('Invalid size')

    return Matrix([[0] * size for _ in range(size)])


# создать единичную матрицу
def eye(size: int) -> Matrix:
    if not size > 0:
        raise ValueError('Invalid size')

    result = zeros(size)

    for i in range(size):
        result[i, i] = 1

    return result


# расширить матрицу единицами
def extend(a: Matrix, size: int) -> Matrix:
    if not size > 0 or len(a) > size:
        raise ValueError('Invalid size')

    if not a.square:
        raise ValueError('Row extend in not supported')

    n = len(a)
    result = zeros(size)
    e_size = size - n

    for i in range(e_size):
        result[i, i] = 1

    for i in range(n):
        for j in range(n):
            result[e_size + i, e_size + j] = a[i, j]

    return result


# убрать нули главной диагонали
def permutation(a: Matrix, b: Vector) -> Tuple:
    p = a.permutation

    return p * a, p * b


# получить вектор по индексу
def column(a: Matrix, j: int) -> Vector:
    if j < 0 or len(a) <= j:
        raise IndexError('Invalid index')

    n = len(a)
    result = Vector([0 for _ in range(n)])

    for i in range(len(a)):
        result[i] = a[i, j]

    return result
