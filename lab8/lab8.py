import numpy as np
import scipy.stats as sps

def pearson_(x, y):
    """
    Расчёт коэффициента корреляции Пирсона
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Выборки должны быть одинаковой длины")
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x)**2 for xi in x)
    denominator_y = sum((yi - mean_y)**2 for yi in y)
    
    if denominator_x == 0 or denominator_y == 0:
        r = 0.0
    else:
        r = numerator / np.sqrt(denominator_x * denominator_y)
    
    if r == 1 or r == -1:
        p_value = 0.0
    else:
        t_value = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
        p_value = 2 * (1 - sps.t.cdf(abs(t_value), df=n-2))
    
    return r, p_value

def spearman_(x, y):
    """
    Расчёт коэффициента корреляции Спирмена
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Выборки должны быть одинаковой длины")
    
    def compute_ranks(data):
        ranked = {}
        sorted_data = sorted((val, i) for i, val in enumerate(data))
        i = 0
        while i < n:
            j = i
            while j < n and sorted_data[j][0] == sorted_data[i][0]:
                j += 1
            
            rank = (i + 1 + j) / 2.0
            
            for k in range(i, j):
                ranked[sorted_data[k][1]] = rank
            i = j
        return [ranked[i] for i in range(n)]
    
    rank_x = compute_ranks(x)
    rank_y = compute_ranks(y)
    
    return pearson_(rank_x, rank_y)

def task1():
    x_values = []
    y_values = []
    
    x_bins = [100, 105, 110, 115, 120, 125]
    y_bins = [35, 45, 55, 65, 75]
    frequencies = [
        [4, 0, 6, 7, 8, 3],
        [5, 5, 2, 10, 0, 0],
        [6, 7, 0, 0, 2, 3],
        [0, 6, 5, 4, 0, 2],
        [5, 1, 2, 4, 3, 0]
    ]
    
    for i in range(len(y_bins)):
        for j in range(len(x_bins)):
            for _ in range(frequencies[i][j]):
                x_values.append(x_bins[j])
                y_values.append(y_bins[i])
    
    r_pearson, p_pearson = pearson_(x_values, y_values)
    rho_spearman, p_spearman = spearman_(x_values, y_values)
    
    print("\nЗадача 1:")
    print(f"Коэффициент корреляции Пирсона: r = {r_pearson}, p = {p_pearson}")
    print(f"Коэффициент корреляции Спирмена: rho = {rho_spearman}, p = {p_spearman}")
    r_lib, p_lib = sps.pearsonr(x_values, y_values)
    rho_lib, p_rho_lib = sps.spearmanr(x_values, y_values)
    print("Scipy (Пирсон):", r_lib, p_lib)
    print("Scipy (Спирмен):", rho_lib, p_rho_lib)
    shapiro_test_x = sps.shapiro(x_values)
    shapiro_test_y = sps.shapiro(y_values)
    print(f"Проверка на нормальность: \n(для x) W = {shapiro_test_x.statistic} и p = {shapiro_test_x.pvalue:.10f}, \n(для у) W = {shapiro_test_y.statistic} и p = {shapiro_test_y.pvalue:.10f}\n")

def task2():
    teacher1 = [98, 94, 88, 80, 76, 70, 63, 61, 60, 58, 56, 51]
    teacher2 = [99, 91, 93, 74, 78, 65, 64, 66, 52, 53, 48, 62]
    
    r_pearson, p_pearson = pearson_(teacher1, teacher2)
    rho_spearman, p_spearman = spearman_(teacher1, teacher2)
    
    print("\nЗадача 2:")
    print(f"Коэффициент корреляции Пирсона: r = {r_pearson}, p = {p_pearson:.10f}")
    print(f"Коэффициент корреляции Спирмена: rho = {rho_spearman}, p = {p_spearman:.10f}")
    r_lib, p_lib = sps.pearsonr(teacher1, teacher2)
    rho_lib, p_rho_lib = sps.spearmanr(teacher1, teacher2)
    print(f"Scipy (Пирсон): r = {r_lib}, p = {p_lib:.10f}")
    print(f"Scipy (Спирмен): rho = {rho_lib}, p = {p_rho_lib:.10f}")
    shapiro_test_x = sps.shapiro(teacher1)
    shapiro_test_y = sps.shapiro(teacher2)
    print(f"Проверка на нормальность: \n(для учителя 1) W = {shapiro_test_x.statistic} и p = {shapiro_test_x.pvalue}, \n(для учителя 2) W = {shapiro_test_y.statistic} и p ={shapiro_test_y.pvalue}\n")

task1()
task2()
