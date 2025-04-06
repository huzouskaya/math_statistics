import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

random_state = 9

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

def empiricalDistributionFunction(data, x, alpha=0.05):
    """
    Вычисляет эмпирическую функцию распределения и 95% доверительный интервал.
    
    :param data: массив данных
    :param x: значение, для которого вычисляется ЭФР
    :param alpha: уровень значимости
    :return: значение ЭФР и доверительный интервал
    """
    n = len(data)
    edf_value = np.sum(data <= x) / n
    error_ = np.sqrt((1/(2*n) * np.log(2/alpha)) / n) 
    
    bottom_ = max(0, edf_value - error_)
    top_ = min(1, edf_value + error_)
    
    return edf_value, (bottom_, top_)

def makePlot(label, data):
    """
    Создает график для эмпирической функции распределения и истинной функции распределения.
    Учитывает, что для дискретных распределений Бернулли и Биномиального не применяется построение через plot
    
    :param label: название распределения
    :param data: массив данных
    """
    x_values = np.sort(data)
    true_cdf = None
    
    if 'U_' in label:
        true_cdf = stats.uniform.cdf(x_values, loc=3, scale=7)
    elif 'Bernoulli' in label:
        true_cdf = stats.bernoulli.cdf(np.arange(0, 2), p=0.2)
        true_cdf = np.array([stats.bernoulli.cdf(x, p=0.2) for x in x_values])
    elif 'Binom' in label:
        true_cdf = np.array([stats.binom.cdf(k, n=20, p=0.3) for k in x_values])
    elif 'Norm' in label:
        true_cdf = stats.norm.cdf(x_values, loc=15, scale=4)
    
    empirical_cdf = [empiricalDistributionFunction(data, x, label)[0] for x in x_values]
    
    confidence_intervals = [empiricalDistributionFunction(data, x, label)[1] for x in x_values]
    bottoms_ = [ci[0] for ci in confidence_intervals]
    tops_ = [ci[1] for ci in confidence_intervals]
    
    plt.figure()

    plot_params = {
        'label': ['Истинная функция распределения', 'Эмпирическая функция распределения', '95% доверительный интервал'],
        'color': ['#0e9c05', '#a304e3', '#cdeee3'],
        'linewidth': [2, 1.5, 1]
    }
    
    if ("Bernoulli" or "Binom_") in label:
        plt.step(x_values, true_cdf, label=plot_params['label'][0], color=plot_params['color'][0], linewidth=plot_params['linewidth'][0])
    else:  
        plt.plot(x_values, true_cdf, label=plot_params['label'][0], color=plot_params['color'][0], linewidth=plot_params['linewidth'][0])
    
    plt.step(x_values, empirical_cdf, label=plot_params['label'][1], color=plot_params['color'][1], linewidth=plot_params['linewidth'][1], where='post')
    plt.fill_between(x_values, bottoms_, tops_, label=plot_params['label'][2], color=plot_params['color'][2], linewidth=plot_params['linewidth'][2], alpha=0.5)

    plt.title(label)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid()
    plt.show()

for label, data in samples_.items():
    makePlot(label, data)