import numpy as np
from scipy import stats

def mhtest(sample, mu0, alternative='two-sided', std=None):
    """
    Проверка гипотезы о среднем
    
    Параметры:
    sample - выборка
    mu0 - гипотетическое среднее
    alternative - 'less', 'greater', 'two-sided'
    std - стандартное отклонение (если известно, для Z-теста)
    
    Возвращает:
    p-value
    """
    n = len(sample)
    sampleMean = sum(sample) / n
    
    if std is not None:
        standartError = std / np.sqrt(n)
        stat = (sampleMean - mu0) / standartError
        
        if alternative == 'greater':
            return 1 - stats.norm.cdf(stat)
        elif alternative == 'less':
            return stats.norm.cdf(stat)
        elif alternative == 'two-sided':
            return 2 * (1 - stats.norm.cdf(abs(stat)))
    else:
        sample_var = sum((x - sampleMean)**2 for x in sample) / (n - 1)
        standartError = np.sqrt(sample_var / n)
        stat = (sampleMean - mu0) / standartError
        
        if alternative == 'greater':
            return 1 - stats.t.cdf(stat, n-1)
        elif alternative == 'less':
            return stats.t.cdf(stat, n-1)
        elif alternative == 'two-sided':
            return 2 * (1 - stats.t.cdf(abs(stat), n-1))

if __name__ == "__main__":
    # mu0 = 0.50         
    # sample_mean = 0.53
    # H1 = 'greater'
    # sigma = 0.11       
    # n = 121             
    # alpha = 0.01 
    # random_state = 9
    # data = stats.norm.rvs(loc=sample_mean, scale=sigma, size=n, random_state=random_state)
    # result = mhtest(sample=data, mu0=mu0, std=sigma, alternative=H1)
    # print(result)
    
    mu0 = 35
    H1 = 'two-sided'
    alpha = 0.05
    values = [34.8, 34.9, 35.0, 35.1, 35.3]
    counts = [2, 3, 4, 6, 5]
    data = np.repeat(values, counts)
    result = mhtest(sample=data, mu0=mu0, alternative=H1, alpha=alpha)
    resultOfScipy = stats.ttest_1samp(data, mu0)
    print(result)
    print(resultOfScipy)
    
    if (result < alpha):
        print("Нулевая гипотеза отклоняется")
    else:
        print("Нулевая гипотеза принимается")