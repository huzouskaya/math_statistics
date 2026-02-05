import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

random_state = 9

# Выборки из требуемых распределений (Равномерное, Бернулли, Биномиальное, Нормальное)
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

# Функция для выборочной средней
def sampleAverage(data):
    return sum(data) / len(data)

# Функция для дисперсии для несмещённой оценки
def varianceDef(data):
    avg_ = sampleAverage(data)
    variance_ = sum((x - avg_) ** 2 for x in data) / (len(data) - 1)
    return variance_

# Функция для дисперсии для смещённой оценки
def varianceDef2(data):
    avg_ = sampleAverage(data)
    variance_ = sum((x - avg_) ** 2 for x in data) / (len(data))
    return variance_

# Функция для стандартного отклонения
def standartDeviation(data):
    return varianceDef(data) ** 0.5

results_ = []

for name, data in samples_.items():
    mean_cust_ = sampleAverage(data)
    var_cust_ = varianceDef(data)
    std_cust_ = standartDeviation(data)
    
    mean_numpy = np.mean(data)
    var_numpy = np.var(data, ddof=1)
    std_numpy = np.std(data, ddof=1)
    
    results_.append({
        'Sample': name,
        'Mean (Custom)': mean_cust_,
        'Variance (Custom)': var_cust_,
        'Standart Dev (Custom)': std_cust_,
        'Mean (Numpy)': mean_numpy,
        'Variance (Numpy)': var_numpy,
        'Standart Deviation (Numpy)': std_numpy,
    })

results_dataframe_ = pd.DataFrame(results_)

print("Results:")
print(results_dataframe_)

for name, data in samples_.items():
    plt.figure(figsize=(10, 6))
    
    plt.hist(data, bins=30, density=True, alpha=0.5, color='g', label='Histogram')
    
    if 'U' in name:
        x = np.linspace(0, 10, 100)
        plt.plot(x, stats.uniform.pdf(x, loc=3, scale=7), 'r-', label='PDF')
    elif 'Bernoulli' in name:
        x = [0, 1]
    elif 'Binom' in name:
        x = np.arange(0, 21)
    elif 'Norm' in name:
        x = np.linspace(5, 25, 100)
        plt.plot(x, stats.norm.pdf(x, loc=15, scale=4), 'r-', label='PDF')
    
    plt.title(name)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()
