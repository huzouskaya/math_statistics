import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.ExcelFile('C:/Users/AliceWolf13/Documents/mathstats/last_chance_for_4/psydata.xlsx')

socio = pd.read_excel(data, 'Социофобия')
attach = pd.read_excel(data, 'ПривязанностьМ')
anxiety = pd.read_excel(data, 'Тревожность')
negativity = pd.read_excel(data, 'Негатив')
fears = pd.read_excel(data, 'Страхи')

df = socio.merge(attach, on='id', suffixes=('_socio', '_attach'))
df = df.merge(anxiety, on='id', suffixes=('', '_anxiety'))
df = df.merge(negativity, on='id', suffixes=('', '_neg'))
df = df.merge(fears, on='id', suffixes=('', '_fears'))

df = df.rename(columns={
    'Сумма': 'Сумма негатива',
    'Сумма_fears': 'Сумма страхов'
})

df['attachment'] = np.where(df['н'] > 5, 'secure', 'insecure')
secure = df[df['attachment'] == 'secure']
insecure = df[df['attachment'] == 'insecure']

def check_normality(data, alpha=0.01):
    if len(data) <= 30:
        stat, p = sps.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = sps.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        test_name = "Kolmogorov-Smirnov"
    return p, test_name

def test_hypothesis(data1, data2, col_name, alpha=0.01):
    print(f"\nАнализ переменной: {col_name}")
    
    p1, test_name1 = check_normality(data1[col_name])
    p2, test_name2 = check_normality(data2[col_name])
    
    print(f"Тест на нормальность ({test_name1}): secure p-value={p1:.4f}")
    print(f"Тест на нормальность ({test_name2}): insecure p-value={p2:.4f}")
    
    if p1 > alpha and p2 > alpha:
        if p1 > alpha and p2 > alpha:
            var1 = np.var(data1[col_name], ddof=1)
            var2 = np.var(data2[col_name], ddof=1)
            f_value = max(var1, var2) / min(var1, var2)
            df1 = len(data1[col_name]) - 1
            df2 = len(data2[col_name]) - 1
            p_var = 2 * min(sps.f.cdf(f_value, df1, df2), 1 - sps.f.cdf(f_value, df1, df2))
            equal_var = p_var > alpha
            print(f"Fisher's test p-value: {p_var:.4f}")
        
        if equal_var:
            t_stat, p_val = sps.ttest_ind(data1[col_name], data2[col_name], equal_var=True, alternative='greater')
            test_name = "Student's t-test"
        else:
            t_stat, p_val = sps.ttest_ind(data1[col_name], data2[col_name], equal_var=False, alternative='greater')
            test_name = "Welch's t-test"
    else:
        u_stat, p_val = sps.mannwhitneyu(data1[col_name], data2[col_name], alternative='greater')
        test_name = "Mann-Whitney U"
        
        h_stat, p_val_kw = sps.kruskal(data1[col_name], data2[col_name])
        print(f"Kruskal-Wallis p-value: {p_val_kw:.4f}")
    
    print(f"Использован тест: {test_name}")
    print(f"p-value: {p_val:.4f}")
    
    if p_val < alpha:
        print(f"Отвергаем H0: статистически значимые различия обнаружены (p={p_val:.4f}). "
            f"Уровень {col_name} значимо выше у ненадёжно привязанных.")
    else:
        print(f"Не отвергаем H0: статистически значимых различий не обнаружено (p={p_val:.4f}). "
            f"Уровень {col_name} не отличается между группами.")

# 1. Проверка гипотезы о социофобии
# Уровень социофобии у ненадёжно привязанных к матери выше, чем у надёжно
test_hypothesis(insecure, secure, 'Общий')

# 2. Проверка гипотез о тревожности
# Уровень тревожности (общей, школьной, самооценочной, межличностной) у ненадёжно привязанных к матери выше, чем у надёжно
for a_type in ['Общая', 'Школьная', 'Самооценочная', 'Межличностная']:
    test_hypothesis(insecure, secure, a_type)

# 3. Проверка гипотезы о страхе негативной оценки
# Уровень страха негативной оценки у ненадёжно привязанных к матери выше, чем у надёжно
test_hypothesis(insecure, secure, 'Сумма негатива')

# 4. Проверка гипотезы об общем балле страхов
# Общий балл страхов у ненадёжно привязанных к матери выше, чем у надёжно
test_hypothesis(insecure, secure, 'Сумма страхов')

# 5. Корреляционный анализ
# У какой из групп больше корреляционных связей между общей тревожностью и каждым из страхов?
fear_columns = [col for col in fears.columns if col not in ['id', 'Сумма']]

def count_correlations(group, columns, alpha=0.01):
    count = 0
    count2 = 0
    for col in columns:
        if col in group.columns:
            try:
                r, p = sps.pearsonr(group['Общая'], group[col])
                if p < alpha:
                    count += 1
                r, p = sps.spearmanr(group['Общая'], group[col])
                if p < alpha:
                    count2 += 1
            except:
                continue
    return count, count2

secure_corr = count_correlations(secure, fear_columns)
insecure_corr = count_correlations(insecure, fear_columns)

def calculate_correlations(group):
    return {col: sps.spearmanr(group['Общая'], group[col]) 
            for col in fear_columns if col in group.columns}
    
def prepare_data(group):
    corrs = calculate_correlations(group)
    return pd.DataFrame.from_dict(
        {k: {'r': v[0], 'p': v[1]} for k, v in corrs.items()}
    ).T

secure_data = prepare_data(secure)
insecure_data = prepare_data(insecure)

corr_data = pd.DataFrame({
    'Надёжная': secure_data['r'],
    'Ненадёжная': insecure_data['r']
})

p_values = pd.DataFrame({
    'Надёжная': secure_data['p'],
    'Ненадёжная': insecure_data['p']
})

plt.figure(figsize=(10, 4))
sns.heatmap(
    corr_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    mask=p_values > 0.01,
    annot_kws={"fontsize": 9}
)
plt.title("Значимые корреляции между страхами и общей тревожностью у исследуемых групп\n")
plt.tight_layout()
plt.show()

print(f"\nКоличество значимых корреляций между общей тревожностью и страхами:")
print(f"Надёжно привязанные: {secure_corr}")
print(f"Ненадёжно привязанные: {insecure_corr}")

if secure_corr > insecure_corr:
    print("Вывод: у надёжно привязанных больше значимых корреляций")
elif insecure_corr > secure_corr:
    print("Вывод: у ненадёжно привязанных больше значимых корреляций")
else:

    print("Вывод: одинаковое количество значимых корреляций")
