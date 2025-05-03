import numpy as np
import pandas as pd
import scipy.stats as stats

params = {
    'U': {'loc': 15, 'scale': 42},
    'Bernoulli': {'p': 0.36},
    'Binom': {'n': 87, 'p': 0.29},
    'Norm': {'loc': 78, 'scale': 20}
}

random_state = 9

samples = {
    'U_100': stats.uniform.rvs(**params['U'], size=100, random_state=random_state),
    'U_1000': stats.uniform.rvs(**params['U'], size=1000, random_state=random_state),
    'Bernoulli_100': stats.bernoulli.rvs(**params['Bernoulli'], size=100, random_state=random_state),
    'Bernoulli_1000': stats.bernoulli.rvs(**params['Bernoulli'], size=1000, random_state=random_state),
    'Binom_100': stats.binom.rvs(**params['Binom'], size=100, random_state=random_state),
    'Binom_1000': stats.binom.rvs(**params['Binom'], size=1000, random_state=random_state),
    'Norm_100': stats.norm.rvs(**params['Norm'], size=100, random_state=random_state),
    'Norm_1000': stats.norm.rvs(**params['Norm'], size=1000, random_state=random_state),
}

def my_bootstrap(data, statistic_func, confidence_level=0.95, n_iterations=1000):
    stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats.append(statistic_func(sample))
    
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(stats, alpha * 100)
    upper = np.percentile(stats, (1 - alpha) * 100)
    return (lower, upper)


def estimate_uniform_loc(data):
    """Оценка параметра loc (минимума) для равномерного распределения"""
    return np.min(data)

def estimate_uniform_scale(data):
    """Оценка параметра scale (размаха) для равномерного распределения"""
    return np.max(data) - np.min(data)

def estimate_bernoulli_p(data):
    """Оценка параметра p (вероятности успеха)"""
    return np.mean(data)

def estimate_binom_n(data):
    """Оценка параметра n (числа испытаний) методом моментов"""
    m1 = np.mean(data)
    m2 = np.mean(np.square(data))
    n_est = m1 / (1 - (m2 - m1**2) / m1) if m1 != 0 else np.max(data)
    return max(np.ceil(n_est), np.max(data))

def estimate_binom_p(data):
    """Оценка параметра p (вероятности успеха)"""
    n_est = estimate_binom_n(data)
    if n_est == 0 or len(data) == 0:
        return 0.0
    return np.clip(np.mean(data) / n_est, 0.0, 1.0)

def estimate_normal_mean(data):
    return np.mean(data)

def estimate_normal_std(data):
    return np.std(data, ddof=1)

results = []

for name, data in samples.items():
    if 'U_' in name:
        loc_low, loc_up = my_bootstrap(data, estimate_uniform_loc)
        scale_low, scale_up = my_bootstrap(data, estimate_uniform_scale)
        
        res_loc = stats.bootstrap((data,), estimate_bernoulli_p, method='percentile')
        res_scale = stats.bootstrap((data,), estimate_bernoulli_p, method='percentile')
        
        results.extend([
            {'Распределение': name, 'Параметр': 'loc', 'Истинное значение':  params['U']['loc'], 'Нижняя граница ДИ': loc_low, 'Верхняя граница ДИ': loc_up, 'Scipy Stats ДИ': res_loc.confidence_interval},
            {'Распределение': name, 'Параметр': 'scale', 'Истинное значение':  params['U']['scale'], 'Нижняя граница ДИ': scale_low, 'Верхняя граница ДИ': scale_up, 'Scipy Stats ДИ': res_scale.confidence_interval}
        ])
    
    elif 'Bernoulli_' in name:
        p_low, p_up = my_bootstrap(data, estimate_bernoulli_p)
        
        res_p = stats.bootstrap((data,), estimate_bernoulli_p, method='percentile')
        
        results.append({
            'Распределение': name, 'Параметр': 'p', 'Истинное значение': params['Bernoulli']['p'], 'Нижняя граница ДИ': p_low, 'Верхняя граница ДИ': p_up,  'Scipy Stats ДИ': res_p.confidence_interval
        })
    
    elif 'Binom_' in name:
        k_low, k_up = my_bootstrap(data, estimate_binom_n)
        p_low, p_up = my_bootstrap(data, estimate_binom_p)
        
        res_k = stats.bootstrap((data,), estimate_binom_n, method='percentile')
        res_p = stats.bootstrap((data,), estimate_binom_p, method='percentile')
        
        results.extend([
            {'Распределение': name, 'Параметр': 'k', 'Истинное значение': params['Binom']['n'], 'Нижняя граница ДИ': k_low, 'Верхняя граница ДИ': k_up, 'Scipy Stats ДИ': res_k.confidence_interval},
            {'Распределение': name, 'Параметр': 'p', 'Истинное значение': params['Binom']['p'], 'Нижняя граница ДИ': p_low, 'Верхняя граница ДИ': p_up, 'Scipy Stats ДИ': res_p.confidence_interval}
        ])
    
    elif 'Norm_' in name:
        mean_low, mean_up = my_bootstrap(data, estimate_normal_mean)
        std_low, std_up = my_bootstrap(data, estimate_normal_std)
        
        res_mean = stats.bootstrap((data,), estimate_normal_mean, method='percentile')
        res_std = stats.bootstrap((data,), estimate_normal_std, method='percentile')
        
        results.extend([
            {'Распределение': name, 'Параметр': 'mean', 'Истинное значение': params['Norm']['loc'], 'Нижняя граница ДИ': mean_low, 'Верхняя граница ДИ': mean_up, 'Scipy Stats ДИ': res_mean.confidence_interval},
            {'Распределение': name, 'Параметр': 'std', 'Истинное значение': params['Norm']['scale'], 'Нижняя граница ДИ': std_low, 'Верхняя граница ДИ': std_up, 'Scipy Stats ДИ': res_std.confidence_interval}
        ])


results_df = pd.DataFrame(results)
print('\n')
print("Оценка параметров распределений:\n")
print(results_df[['Распределение', 'Параметр', 'Истинное значение', 'Нижняя граница ДИ', 'Верхняя граница ДИ', 'Scipy Stats ДИ']].to_string(index=False))
print('\n')