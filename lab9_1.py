import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
import scipy.stats as sps

def load_data():
    data = np.loadtxt('C:/Users/AliceWolf13/Documents/mathstats/pain9/regressia.txt', delimiter=';')
    df = pd.DataFrame({'y': data})
    df['x'] = df.index
    return df

def plot_data(x, y, title, color='blue', label='Данные', pred=None, pred_label=None, pred_color=None):
    plt.scatter(x, y, color=color, label=label)
    if pred is not None:
        plt.plot(x, pred, color=pred_color, label=pred_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def check_gauss_markov(model, X, model_name):
    print(f"\nПроверка условий теоремы Гаусса-Маркова для {model_name}:")
    residuals = model.resid
    
    shapiro_test = sps.shapiro(residuals)
    print_result("1.1.1 Тест на нормальность остатков (Шапиро-Уилк):", 
                shapiro_test[1] > 0.05, 
                f"Статистика: {shapiro_test[0]:.4f}, p-value = {shapiro_test[1]:.4f}")
    
    if len(residuals) > 30:
        ks_test = sps.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
        print_result("1.1.2 Тест Колмогорова-Смирнова:",
            ks_test[1] > 0.05,
            f"Статистика: {ks_test[0]:.4f}, p-value = {ks_test[1]:.4f}")
    
    bp_test = het_breuschpagan(residuals, X)
    print_result("\n1.2 Тест на неоднородность дисперсии (Бройша-Пагана):", 
                bp_test[1] > 0.05,
                f"Статистика: {bp_test[0]:.4f}, p-value = {bp_test[1]:.4f}")
    
    t_stat, p_value = sps.ttest_1samp(residuals, 0)
    print_result("\n1.3 Проверка нулевого матожидания остатков:", 
                p_value > 0.05,
                f"t-статистика: {t_stat:.4f}, p-value = {p_value:.4f}")
    
    bg_test = acorr_breusch_godfrey(model, nlags=1)
    print_result("\n2. Тест на автокорреляцию (Бройша-Годфри):", 
                bg_test[1] > 0.05,
                f"p-value = {bg_test[1]:.4f}")

    rank = np.linalg.matrix_rank(X)
    print(f"\n3. Проверка полного ранга матрицы X:")
    print(f"    Ранг матрицы X: {rank}, число параметров m: {model.df_model + 1}")
    print("    Условие rang(X) = m выполняется" if rank == model.df_model + 1 else 
        "    Столбцы X линейно зависимы, условие теоремы не выполняется")

def print_result(test_name, condition, stats):
    print(f"{test_name}\n    {stats}")
    print("    Условие выполняется" if condition else "    Условие не выполняется")

def main():
    df = load_data()
    plot_data(df['x'], df['y'], 'Диаграмма рассеяния')
    
    X_linear = sm.add_constant(df['x'])
    model_linear = sm.OLS(df['y'], X_linear).fit()
    print(f"Линейная модель:\ny = {model_linear.params['x']:.6f}x + {model_linear.params['const']:.6f}")
    plot_data(df['x'], df['y'], 'Линейная регрессия', pred=model_linear.predict(X_linear), 
            pred_label='Линейная регрессия', pred_color='red')
    check_gauss_markov(model_linear, X_linear, "линейной модели")
    print(model_linear.summary())
    
    for d in range(2, 4):
        df[f'x^{d}'] = df['x']**d
    X_poly = sm.add_constant(df[['x', 'x^2', 'x^3']])
    model_poly = sm.OLS(df['y'], X_poly).fit()
    print("\nПолиномиальная модель 3-й степени:")
    equation = f"y = {model_poly.params['const']:.6f}"
    for d in range(1, 4):
        equation += f" + {model_poly.params[f'x^{d}' if d>1 else 'x']:.6f}x^{d}" if d>1 else f" + {model_poly.params['x']:.6f}x"
    print(equation)
    plot_data(df['x'], df['y'], 'Полиномиальная регрессия 3-й степени', 
            pred=model_poly.predict(X_poly), pred_label='Полином 3-й степени', pred_color='green')
    check_gauss_markov(model_poly, X_poly, "полиномиальной модели 3-й степени")
    print(model_poly.summary())
    
    plt.scatter(df['x'], df['y'], color='blue', s=50, label='Данные')
    plt.plot(df['x'], model_linear.predict(X_linear), color='red', label='Линейная')
    plt.plot(df['x'], model_poly.predict(X_poly), color='green', label='Полином 3-й степени')
    plt.title('Сравнение регрессионных моделей')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()