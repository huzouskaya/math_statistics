import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Установка начального состояния генератора случайных чисел для воспроизводимости
random_state = 9

# Генерация выборок из различных распределений
samples_ = {
    'U_100': stats.uniform.rvs(loc=3, scale=7, size=100, random_state=random_state),
    'U_1000': stats.uniform.rvs(loc=3, scale=7, size=1000, random_state=random_state),
    'Bernoulli_100': stats.bernoulli.rvs(p=0.2, size=100, random_state=random_state),
    'Bernoulli_1000': stats.bernoulli.rvs(p=0.2, size=1000, random_state=random_state),
    'Binom_100': stats.binom.rvs(n=20, p=0.3, size=100, random_state=random_state),
    'Binom_1000': stats.binom.rvs(n=20, p=0.3, size=1000, random_state=random_state),
    'Norm_100': stats.norm.rvs(loc=15, scale=4, size=100, random_state=random_state),
    'Norm_1000': stats.norm.rvs(loc=15, scale=4, size=1000, random_state=random_state),
}

def critical_value(label, alpha):
    """
    Вычисляет критическое значение для различных распределений.
    
    :param data: массив данных
    :param alpha: уровень значимости
    :return: критическое значение
    """
    if 'Bernoulli' in label:
        return stats.bernoulli.ppf(1 - alpha, p=0.2)
    elif 'Binom' in label:
        return stats.binom.ppf(1 - alpha, n=20, p=0.3)
    elif 'U_' in label:
        return stats.uniform.ppf(1 - alpha, loc=3, scale=7)
    elif 'Norm' in label:
        return stats.norm.ppf(1 - alpha / 2)  # Для нормального распределения
    else:
        raise ValueError("Неизвестное распределение")

def empiricalDistributionFunction(data, x, label, alpha=0.05):
    """
    Вычисляет эмпирическую функцию распределения и 95% доверительный интервал.
    
    :param data: массив данных
    :param x: значение, для которого вычисляется ЭФР
    :param label: название распределения
    :param alpha: уровень значимости
    :return: значение ЭФР и доверительный интервал
    """
    n = len(data)
    efd_value = np.sum(data <= x) / n  # Эмпирическая функция распределения
    
    z = critical_value(label, alpha)  # Критическое значение для соответствующего распределения
    
    # Вычисление стандартной ошибки
    error_ = z * np.sqrt((efd_value * (1 - efd_value)) / n) 
    
    # Определение границ доверительного интервала
    bottom_ = max(0, efd_value - error_)
    top_ = min(1, efd_value + error_)
    
    return efd_value, (bottom_, top_)

def makePlot(label, data):
    """
    Создает график для эмпирической функции распределения и истинной функции распределения.
    
    :param label: название распределения
    :param data: массив данных
    """
    x_values = np.sort(data)  # Сортировка данных
    true_cdf = None
    
    if 'U_' in label:
        true_cdf = stats.uniform.cdf(x_values, loc=3, scale=7)
    elif 'Bernoulli' in label:
        # Для бернуллиевского распределения
        true_cdf = stats.bernoulli.cdf(np.arange(0, 2), p=0.2)
        true_cdf = np.array([stats.bernoulli.cdf(x, p=0.2) for x in x_values])
    elif 'Binom' in label:
        # Для биномиального распределения
        true_cdf = np.array([stats.binom.cdf(k, n=20, p=0.3) for k in x_values])
    elif 'Norm' in label:
        true_cdf = stats.norm.cdf(x_values, loc=15, scale=4)
    
    empirical_cdf = [empiricalDistributionFunction(data, x, label)[0] for x in x_values]
    
    # Вычисление доверительных интервалов
    confidence_intervals = [empiricalDistributionFunction(data, x, label)[1] for x in x_values]
    lowers_ = [ci[0] for ci in confidence_intervals]
    uppers_ = [ci[1] for ci in confidence_intervals]
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, true_cdf, label='Истинная функция распределения', color='blue')
    plt.step(x_values, empirical_cdf, label='Эмпирическая функция распределения', color='orange', where='post')
    plt.fill_between(x_values, lowers_, uppers_, color='gray', alpha=0.5, label='95% доверительный интервал')
    
    plt.title(label)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid()
    plt.show()

# Генерация графиков для всех выборок
for label, data in samples_.items():
    makePlot(label, data)