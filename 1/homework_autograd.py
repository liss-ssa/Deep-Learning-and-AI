import torch

# 2.1 Создайте тензоры x, y, z с requires_grad=True
def task1():
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)   
    z = torch.tensor(3.0, requires_grad=True)

    # Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
    f = x**2 + y**2 + z**2 + 2*x*y*z    

    # Найдите градиенты по всем переменным
    f.backward()

    # Выводим градиенты
    print("Задание 2.1:")
    print(f"Градиент по x: {x.grad.item()}")
    print(f"Градиент по y: {y.grad.item()}")
    print(f"Градиент по z: {z.grad.item()}")

    # Проверьте результат аналитически
    df_dx_analytical = 2*x.item() + 2*y.item()*z.item()
    df_dy_analytical = 2*y.item() + 2*x.item()*z.item()
    df_dz_analytical = 2*z.item() + 2*x.item()*y.item()

    print("\nАналитические результаты:")
    print(f"df/dx: {df_dx_analytical}")
    print(f"df/dy: {df_dy_analytical}")
    print(f"df/dz: {df_dz_analytical}")

    return x.grad, y.grad, z.grad


# 2.2 Реализуйте функцию MSE (Mean Squared Error):
def task2():
    # MSE = (1/n) * Σ(y_pred - y_true)^2
    w = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)   
    # где y_pred = w * x + b (линейная функция)
    # Найдите градиенты по w и b
    # Создаем данные
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])  # Идеальная зависимость y = 2x

    y_pred = w * x + b

    MSE = torch.mean((y_pred - y_true)**2)

    # Обратное распространение
    MSE.backward()

    # Вывод градиентов
    print(f"Градиент по w: {w.grad.item()}")
    print(f"Градиент по b: {b.grad.item()}")

    # Аналитическая проверка
    n = len(x)
    dw_analytical = (2/n) * torch.sum((y_pred - y_true) * x)
    db_analytical = (2/n) * torch.sum(y_pred - y_true)

    print("Задание 2.2:")
    print(f"MSE: {MSE.item():}")
    print(f"dw: autograd={w.grad.item():}, analytical={dw_analytical.item():}")
    print(f"db: autograd={b.grad.item():}, analytical={db_analytical.item():}\n")

    return w.grad, b.grad


# 2.3 Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad
def task3():
    x = torch.tensor(1.5, requires_grad=True)

    # Вычисляем функцию
    f = torch.sin(x**2 + 1)

    # Вычисляем градиент
    f.backward(retain_graph=True)
    autograd_result = x.grad.item()

    # Проверка с помощью torch.autograd.grad
    grad_check = torch.autograd.grad(f, x)[0]
    x.grad.zero_()
    # Аналитический градиент
    df_dx_analytical = torch.cos(x**2 + 1) * 2*x

    print("Задание 2.3:")
    print(f"1. backward(): {autograd_result}")
    print(f"2. autograd.grad(): {grad_check.item()}") 
    print(f"Аналитический результат: {df_dx_analytical.item()}")

    return autograd_result

if __name__ == "__main__":
    task1()
    task2()
    task3()

