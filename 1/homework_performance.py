import torch
import time
from tabulate import tabulate


# 3.1 Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами

def create_matrix():
    sizes = [
        (64, 1024, 1024),
        (128, 512, 512),
        (256, 256, 256)
    ]
    
    # Проверка допустимости размеров
    for size in sizes:
        if any(dim <= 0 for dim in size):
            raise ValueError(f"Недопустимый размер матрицы: {size}. Все размерности должны быть положительными.")
    
    try:
        cpu_matrices = [torch.randn(size, device='cpu', dtype=torch.float32) for size in sizes]
        
        if torch.cuda.is_available():
            cuda_matrices = [mat.cuda() for mat in cpu_matrices]
            return cpu_matrices, cuda_matrices
        return cpu_matrices, None
        
    except RuntimeError as e:
        raise RuntimeError(f"Ошибка создания матриц: {str(e)}")


# 3.2 Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU

def measure_time(operation, *args, device='cpu'):
    try:
        if device == 'cuda' and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            result = operation(*args)
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000  # в секунды
        else:
            start_time = time.time()
            result = operation(*args)
            elapsed_time = time.time() - start_time
        
        return result, elapsed_time
    
    except Exception as e:
        raise RuntimeError(f"Ошибка при выполнении операции: {str(e)}")


# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде

def comparison_model(cpu_matrices, cuda_matrices):
    operations = {
        'Матричное умножение': lambda x: torch.matmul(x, x),
        'Поэлементное сложение': lambda x: x + x,
        'Поэлементное умножение': lambda x: x * x,
        'Транспонирование': lambda x: x.transpose(-2, -1),
        'Сумма всех элементов': lambda x: x.sum()
    }
    
    results = []
    
    for op_name, op_func in operations.items():
        for i, cpu_mat in enumerate(cpu_matrices):
            size = cpu_mat.size()
            
            # Проверка совместимости размеров для операций
            if op_name == 'Матричное умножение':
                if cpu_mat.dim() != 2 or cpu_mat.size(-1) != cpu_mat.size(-2):
                    continue  # Пропускаем неквадратные матрицы для умножения
            
            # Измерение на CPU
            try:
                _, cpu_time = measure_time(op_func, cpu_mat, device='cpu')
            except RuntimeError as e:
                print(f"Ошибка на CPU для {op_name} {size}: {str(e)}")
                continue
            
            # Измерение на GPU если доступно
            cuda_time = None
            if cuda_matrices is not None:
                try:
                    _, cuda_time = measure_time(op_func, cuda_matrices[i], device='cuda')
                except RuntimeError as e:
                    print(f"Ошибка на GPU для {op_name} {size}: {str(e)}")
                    cuda_time = None
            
            # Вычисление ускорения
            speedup = None
            if cuda_time is not None and cuda_time > 0:
                speedup = cpu_time / cuda_time
                
            results.append({
                'Операция': op_name,
                'Размер': str(size),
                'CPU время (с)': f"{cpu_time:.6f}",
                'CUDA время (с)': f"{cuda_time:.6f}" if cuda_time is not None else "N/A",
                'Ускорение': f"{speedup:.2f}x" if speedup is not None else "N/A"
            })
    
    return results

def print_results(results) -> None:
    print("Результаты сравнения производительности CPU vs CUDA:")
    print(tabulate(results, headers="keys", tablefmt="grid", stralign="right"))

def test_small_example():
    print("Тестирование на маленьких матрицах (3x3)")
    
    cpu_mat = torch.randn(3, 3)
    cuda_mat = cpu_mat.cuda() if torch.cuda.is_available() else None
    
    # Проверка матричного умножения
    cpu_res, _ = measure_time(lambda x: x @ x, cpu_mat, device='cpu')
    if cuda_mat is not None:
        cuda_res, _ = measure_time(lambda x: x @ x, cuda_mat, device='cuda')
        assert torch.allclose(cpu_res, cuda_res.cpu(), atol=1e-6), "Результаты CPU и GPU не совпадают"
    
    print("Все тесты пройдены успешно")

def task3():
    print("Задание 3:")
    
    # 3.1 Подготовка данных
    cpu_matrices, cuda_matrices = create_matrix()
 
    # Тестирование на маленьких данных
    test_small_example()

    if not torch.cuda.is_available():
        print("Внимание: CUDA недоступно, тестирование будет выполнено только на CPU")
    
    # 3.3 Сравнение операций
    results = comparison_model(cpu_matrices, cuda_matrices)
    
    print_results(results)
    
    return results

if __name__ == "__main__":
    task3()