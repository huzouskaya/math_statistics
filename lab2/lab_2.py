# Целью данной лабораторной работы является получение точечных оценок параметров распределений методом моментов и методом максимального правдоподобия.
# Список распределений:

# 0. Bernoulli (p);
# 1. U (a, b);
# 2. N (μ, σ^2);
# 3. Poisson ⁡(λ);
# 4. Bin (k, p);
# 5. Exp (λ).

# +(Poisson) Выбрать распределение под номером, равным вашему номеру в списке группы по модулю 6.
# (Параметр \lambda) Подробно вывести в общем виде методом моментов оценки всех параметров распределения.
# То же самое методом максимального правдоподобия.
# (rvs) Сгенерировать 2 выборки из распределения на 100 и 10000 элементов. Параметры распределения должны быть нетривиальными.
# (DataFrame) Вывести в табличной форме результат. (ориентировочная шапка таблицы: размер выборки, истинные параметры, оценка мм, оценка ммп)
# (Распределение Пуассона не имеет этого метода) Если распределение имеет метод fit, проверить, что оценка максимального правдоподобия методом fit совпадает с полученной.


import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize_scalar

def factorial(n):
    if n < 0:
        raise ValueError("Факториал не определен для отрицательных чисел")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

true_lambda = 5
random_state = 9

samples = {
    'Poisson_100': stats.poisson.rvs(mu=true_lambda, size=100, random_state=random_state),
    'Poisson_10000': stats.poisson.rvs(mu=true_lambda, size=10000, random_state=random_state)
}

results = []

for key, sample in samples.items():
    n = len(sample)
    
    lambda_mm = np.mean(sample)
    
    def negative_logLikelihood(lambda_):
        if lambda_ <= 0:
            return np.inf
        return -(np.sum(sample * np.log(lambda_)) - n * lambda_ - np.sum([np.log(factorial(x)) for x in sample]))

    try:
        result = minimize_scalar(negative_logLikelihood, bounds=(0, 100), method='bounded')
        lambda_mle = result.x
    except Exception as e:
        print(f"Ошибка при минимизации для {key}: {e}")
        lambda_mle = np.nan

    results.append({
        'Размер выборки': n,
        'Истинные параметры (λ)': true_lambda,
        'Оценка ММ (λ)': lambda_mm,
        'Оценка ММП (λ)': lambda_mle,
    })

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False), "\n")
