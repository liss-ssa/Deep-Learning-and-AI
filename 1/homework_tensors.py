import torch

# 1.1 Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor1 = torch.rand(3, 4)

# - Тензор размером 2x3x4, заполненный нулями
tensor2 = torch.zeros(2, 3, 4)

# - Тензор размером 5x5, заполненный единицами
tensor3 = torch.ones(5, 5)

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
tensor4 = torch.arange(16).reshape(4, 4)


# 1.2 Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.rand(3, 4)
B = torch.rand(4, 3)
# Выполните:
# - Транспонирование тензора A
A_transpose = torch.transpose(A, 0, 1)

# - Матричное умножение A и B
A_mul_B = torch.matmul(A, B)

# - Поэлементное умножение A и транспонированного B
elementwise = A * torch.transpose(B, 0, 1)

# - Вычислите сумму всех элементов тензора A
A_sum = torch.sum(A)    


# 1.3 Создайте тензор размером 5x5x5
tensor5 = torch.rand(5, 5, 5)
# Извлеките:
# - Первую строку
first_str = tensor5[0, :, :]

# - Последний столбец
last_col = tensor5[:, :, -1]

# - Подматрицу размером 2x2 из центра тензора
centr_matr = tensor5[1:3, 1:3, 1:3]

# - Все элементы с четными индексами
even = tensor5[::2, ::2, ::2]


# 1.4 Создайте тензор размером 24 элемента
tensor6 = torch.arange(24)
# Преобразуйте его в формы:
# - 2x12
shape1 = tensor6.view(2, 12)

# - 3x8
shape2 = tensor6.view(3, 8)

# - 4x6
shape3 = tensor6.view(4, 6)

# - 2x3x4
shape4 = tensor6.view(2, 3, 4)

# - 2x2x2x3
shape5 = tensor6.view(2, 2, 2, 3)

