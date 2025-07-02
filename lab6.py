import numpy as np
from scipy import stats

def compare_means(sample1=None, sample2=None, var1=None, var2=None, dependent=False, alternative='two-sided'):
    if sample1 is None or sample2 is None:
        raise ValueError("Нужны обе выборки")
    
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    
    if dependent:
        if n1 != n2:
            raise ValueError("Для зависимых выборок размеры должны совпадать")
        diffs = sample1 - sample2
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(n1))
        df = n1 - 1
        return pvalue_(t_stat, df, alternative, 't')
    
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    if var1 is not None and var2 is not None:
        z_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
        return pvalue_(z_stat, None, alternative, 'z')
    
    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1)
    
    pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2)
    t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
    df = n1 + n2 - 2
    
    return pvalue_(t_stat, df, alternative, 't')

def pvalue_(stat, df, alternative, dist_type):
    if dist_type == 't':
        if alternative == 'two-sided':
            return 2 * (1 - stats.t.cdf(abs(stat), df))
        elif alternative == 'less':
            return stats.t.cdf(stat, df)
        else:
            return 1 - stats.t.cdf(stat, df)
    elif alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(abs(stat)))
    elif alternative == 'less':
        return stats.norm.cdf(stat)
    else:
        return 1 - stats.norm.cdf(stat)

if __name__ == "__main__":
    print()
    
    # задача 1
    alpha = 0.05
    sample1 = stats.norm.rvs(loc=130, scale=np.sqrt(60), size=30)
    sample2 = stats.norm.rvs(loc=125, scale=np.sqrt(80), size=40)
    p = compare_means(sample1, sample2, var1=60, var2=80)
    # scipy_st, scipy_p = stats.ttest_ind(sample1, sample2, equal_var=True, alternative='two-sided')
    # print("Задача 1:", p, "scipy_p = ", scipy_p)
    print(f"{'Отвергаем' if p < alpha else 'Принимаем'} H0 при alpha={alpha}")
    
    # задача 2
    alpha = 0.02
    values1 = [3.4, 3.5, 3.7, 3.9]
    counts1 = [2, 3, 4, 1]
    sample1 = np.repeat(values1, counts1)
    values2 = [3.2, 3.4, 3.6]
    counts2 = [2, 2, 8]
    sample2 = np.repeat(values2, counts2)
    p = compare_means(sample1, sample2)
    scipy_st, scipy_p = stats.ttest_ind(sample1, sample2, equal_var=True, alternative='two-sided')
    print("Задача 2: p = ", p, "scipy_p = ", scipy_p)
    print(f"{'Отвергаем' if p < alpha else 'Принимаем'} H0 при alpha={alpha}")
    
    # задача 3
    alpha = 0.05
    sample1=[2, 3, 5, 6, 8, 10]
    sample2=[10, 3, 6, 1, 7, 4]
    p = compare_means(sample1, sample2, dependent=True)
    scipy_st, scipy_p = stats.ttest_rel(sample1, sample2, alternative='two-sided')
    print("Задача 3:", p, "scipy_p = ", scipy_p)
    print(f"{'Отвергаем' if p < alpha else 'Принимаем'} H0 при alpha={alpha}")
    
    print()
    
    # ttest_ind(a, b, *, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0, method=None, keepdims=False)