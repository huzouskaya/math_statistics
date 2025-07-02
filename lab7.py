import numpy as np
import scipy.stats as sps
from statsmodels.distributions.empirical_distribution import ECDF

def kolmogorov_cdf(x):
    """Функция распределения статистики Колмогорова"""
    if x <= 0:
        return 0.0
    k = np.arange(1, 200)
    terms = (-1)**(k-1) * np.exp(-2 * (k**2) * (x**2))
    return 1 - 2 * np.sum(terms)

def kolmogorov_test(sample, cdf):
    """Критерий Колмогорова для проверки соответствия распределению"""
    n = len(sample)
    Dn = np.max(np.abs(np.arange(1, n+1)/n - cdf(np.sort(sample))))
    p_value = 1 - kolmogorov_cdf(np.sqrt(n) * Dn)
    return p_value

def chi2_test(sample, bins, cdf):
    """Критерий хи-квадрат для проверки соответствия распределению"""
    observed, _ = np.histogram(sample, bins=bins)
    n = len(sample)
    expected = n * np.diff(cdf(bins))
    chi2_stat = np.sum((observed - expected)**2 / expected)
    p_value = 1 - sps.chi2.cdf(chi2_stat, df=len(bins)-1)
    return p_value

def homogenity_test(sample1, sample2):
    """Критерий Колмогорова-Смирнова для проверки однородности"""
    n = len(sample1)
    m = len(sample2)
    
    combo_sorted = np.sort(np.concatenate([sample1, sample2]))
    
    ecdf1 = ECDF(sample1)(combo_sorted)
    ecdf2 = ECDF(sample2)(combo_sorted)
    
    Dn = np.max(np.abs(ecdf1 - ecdf2))
    K = np.sqrt((n * m) / (n + m)) * Dn
    p_value = 1 - kolmogorov_cdf(K)
    
    p_lib = sps.ks_2samp(sample1, sample2).pvalue
    return p_value, p_lib

def generate_samples(th_in_group):
    """Генерация выборок по номеру в группе"""
    n = 100
    dist_type = th_in_group % 3
    
    if dist_type == 0:
        samples = [sps.norm.rvs(loc=mu, scale=sigma, size=n) for mu, sigma in [(4, 13), (4, 13), (5, 7)]]
    elif dist_type == 1:
        samples = [sps.uniform.rvs(loc=mu, scale=sigma, size=n) for mu, sigma in [(0, 12), (0, 12), (6, 9)]]
    else:
        samples = [sps.binom.rvs(n=k, p=p, size=n) for k, p in [(13, 0.7), (13, 0.7), (32, 0.5)]]
    
    return samples

def test_hypotheses(sample1, sample2, sample3, dist_type):
    results = {}
    
    if dist_type == 'norm':
        mu, sigma = np.mean(sample1), np.std(sample1)
        p_custom = kolmogorov_test(sample1, lambda x: sps.norm.cdf(x, loc=mu, scale=sigma))
        p_lib = sps.kstest(sample1, 'norm', args=(mu, sigma))[1]
    elif dist_type == 'uniform':
        a = np.min(sample1)
        b = np.max(sample1)
        p_custom = kolmogorov_test(sample1, lambda x: sps.uniform.cdf(x, loc=a, scale=b - a))
        p_lib = sps.kstest(sample1, 'uniform', args=(a, b - a))[1]
    elif dist_type == 'binom':
        p_hat = np.mean(sample1)/13
        bins = np.arange(0, max(sample1)+2) - 0.5
        observed = np.histogram(sample1, bins=bins)[0]
        expected = len(sample1) * np.diff(sps.binom.cdf(bins, n=13, p=p_hat))
        p_custom = chi2_test(sample1, bins, lambda x: sps.binom.cdf(x, n=13, p=0.7))
        p_lib = sps.chisquare(observed, f_exp=expected).pvalue
    results['same_dist'] = {'custom': p_custom, 'library': p_lib}
    
    p_custom_other = kolmogorov_test(sample1, lambda x: sps.uniform.cdf(x, -3, 6))
    p_lib_other = sps.kstest(sample1, 'uniform', args=(-3, 6))[1]
    results['other_dist'] = {'custom': p_custom_other, 'library': p_lib_other}
    
    results['homogenity'] = {
        'sample1 vs sample2': homogenity_test(sample1, sample2),
        'sample1 vs sample3': homogenity_test(sample1, sample3),
        'sample2 vs sample3': homogenity_test(sample2, sample3)
    }
    
    p_shapiro = sps.shapiro(sample1)[1]
    results['normality'] = p_shapiro
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    th_in_group = 9
    
    sample1, sample2, sample3 = generate_samples(th_in_group)
    results = test_hypotheses(sample1, sample2, sample3, 'norm')
    
    # test_values = [0.5, 1.0, 1.5]
    # for x in test_values:
    #     print(f"kolmogorov_cdf({x}) = {kolmogorov_cdf(x)} vs scipy: {sps.kstwobign.cdf(x)}")
    
    print("\nРезультаты проверки гипотез:\n")
    print(f"1. Соответствие своему распределению: custom={results['same_dist']['custom']:.4f}, scipy={results['same_dist']['library']:.4f}")
    print(f"2. Соответствие другому распределению: custom={results['other_dist']['custom']:.4f}, scipy={results['other_dist']['library']:.4f}")
    print("3. Проверка однородности:")
    [print(f"{pair}: кастом={res[0]:.4f} | lib={res[1]:.4f} | Δ={abs(res[0]-res[1])}") 
        for pair, res in results['homogenity'].items()]
    print(f"4. Проверка нормальности (Шапиро-Вилк): p={results['normality']:.4f}", "(нормальность не отвергается)\n\n" if results['normality'] > 0.05 else "(нормальность отвергается)\n\n")