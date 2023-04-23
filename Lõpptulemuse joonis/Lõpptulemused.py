import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy as rp
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols

def determine_grade(score):
    if score > 90 and score <= 100:
        return 'A'
    elif score > 80 and score <= 90:
        return 'B'
    elif score > 70 and score <= 80:
        return 'C'
    elif score > 60 and score <= 70:
        return 'D'
    elif score > 50 and score <= 60:
        return 'E'
    else:
        return 'F ja mitteilmunud'
    

grades = {
    90: "A",
    80: "B",
    70: "C",
    60: "D",
    50: "E",
    -1: "F \n ja mitteilmunud",
}

def grade_mapping(value):
    for key, letter in grades.items():
        if value > key:
            return letter


df = pd.read_csv('/Users/carolin/Documents/Magister/AndmedRühmaga2.csv', decimal=',')
df_copy = df

df_copy = df_copy.replace('-', '0', regex=True)

cols = df.columns
df_copy[cols[1:]] = df_copy[cols[1:]].apply(pd.to_numeric, errors='coerce')

df_copy['aasta'] = df_copy['aasta'].astype(str)

df_copy.rename(columns={'aasta': 'Õppeaasta'}, inplace=True)

# -------------------------------------------------

# tee punktidest tähelised hinded
nimelisedHinded = df_copy['Lõpphinne'].map(grade_mapping)
df_copy["Tulemus"] = pd.Categorical(
    nimelisedHinded, categories=grades.values(), ordered=True
)

# kalkuleeri kui palju neid Asid ja Bsid jne oli 
df_copy['Tulemus'] = df_copy['Tulemus'].astype(str)
percentage = df_copy['Tulemus'].value_counts() * 100

# kalkuleeri kui palju neid Asid ja Bsid jne oli 
grade_counts = (df_copy.groupby(['Õppeaasta'])['Tulemus']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Tulemus'))

count = df_copy.groupby(['Õppeaasta'])['Tulemus'].value_counts().reset_index(name='count')
print(count)


# -------------------------------------------------

sns.set_style("whitegrid", {'grid.color': '.8'})
plt.subplots(figsize=(10, 8))

# tulpdiagramm protsendi ja hinde tulemusega
order = ['2018', '2019', '2020', '2021', '2022.1', '2022.2']

g = sns.barplot(y = 'percentage', x = 'Tulemus', hue='Õppeaasta', data=grade_counts, palette="magma", hue_order=order)
plt.setp(g.collections, alpha=.4)

# pane protsent joonisele veeru peale
for p in g.patches:
    current_width = p.get_width()
    txt = int(p.get_height().round(0))
    if (txt <= 1):
        color = "black"
        txt_y = p.get_height()+0.3
    else:
        color = "w"
        txt_y = p.get_height()-1.6
    
    if (len(str(txt)) == 1):
        txt_x = p.get_x()+0.035
    else:
        txt_x = p.get_x()+0.01
        
    g.text(txt_x, txt_y, txt, color=color, weight='bold')


plt.ylabel("Tulemus (%)", fontweight='bold')
plt.xlabel('Hinded', fontweight='bold')
plt.show()

# --------------------------------------------------------------------------

from scikit_posthocs import posthoc_dunn

data = [df_copy['Lõpphinne'][df_copy['Õppeaasta'] == '2018'],
               df_copy['Lõpphinne'][df_copy['Õppeaasta'] == '2019'],
               df_copy['Lõpphinne'][df_copy['Õppeaasta'] == '2020'],
               df_copy['Lõpphinne'][df_copy['Õppeaasta'] == '2021'],
               df_copy['Lõpphinne'][df_copy['Õppeaasta'] == '2022.1']]

# posthoc dunn test, with correction for multiple testing
dunn_df = posthoc_dunn(data)
# print(dunn_df)

# -----------------------------------------------------------

from scipy.stats import chisquare

first_year =  count[count['Õppeaasta'] == '2018']
second_year = count[count['Õppeaasta'] == '2019']
third_year =  count[count['Õppeaasta'] == '2020']
fourth_year = count[count['Õppeaasta'] == '2021']
fifth_year1 = count[count['Õppeaasta'] == '2022.1']
fifth_year2 = count[count['Õppeaasta'] == '2022.2']

data_cont = pd.crosstab(df_copy['Õppeaasta'], df_copy['Tulemus'])
print(data_cont)
print("\n")

t = stats.chi2_contingency(data_cont)
print(t)
print("\n#############\n")

h1 = stats.chi2_contingency([data_cont.iloc[0][0:6].values, data_cont.iloc[1][0:6].values])
print(h1)
print("\n 2018 & 2019 #############\n")

h2 = stats.chi2_contingency([data_cont.iloc[0][0:6].values, data_cont.iloc[2][0:6].values])
print(h2)
print("\n 2018 & 2020 #############\n")

h3 = stats.chi2_contingency([data_cont.iloc[0][0:6].values, data_cont.iloc[3][0:6].values])
print(h3)
print("\n 2018 & 2021 #############\n")

h4 = stats.chi2_contingency([data_cont.iloc[0][0:6].values, data_cont.iloc[4][0:6].values])
print(h4)
print("\n 2018 & 2022.1 #############\n")

h5 = stats.chi2_contingency([data_cont.iloc[0][0:6].values, data_cont.iloc[5][0:6].values])
print(h5)
print("\n 2018 & 2022.2 #############\n")

h6 = stats.chi2_contingency([data_cont.iloc[1][0:6].values, data_cont.iloc[2][0:6].values])
print(h6)
print(round(h6.pvalue, 4))
print("\n 2019 & 2020 #############\n")

h7 = stats.chi2_contingency([data_cont.iloc[1][0:6].values, data_cont.iloc[3][0:6].values])
print(h7)
print("\n 2019 & 2021 #############\n")

h8 = stats.chi2_contingency([data_cont.iloc[1][0:6].values, data_cont.iloc[4][0:6].values])
print(h8)
print("\n 2019 & 2022.1 #############\n")

h9 = stats.chi2_contingency([data_cont.iloc[1][0:6].values, data_cont.iloc[5][0:6].values])
print(h9)
print("\n 2019 & 2022.2 #############\n")

h10 = stats.chi2_contingency([data_cont.iloc[2][0:6].values, data_cont.iloc[3][0:6].values])
print(h10)
print("\n 2020 & 2021 #############\n")

h11 = stats.chi2_contingency([data_cont.iloc[2][0:6].values, data_cont.iloc[4][0:6].values])
print(h11)
print("\n 2020 & 2022.1 #############\n")

h12 = stats.chi2_contingency([data_cont.iloc[2][0:6].values, data_cont.iloc[5][0:6].values])
print(h12)
print("\n 2020 & 2022.2 #############\n")

h13 = stats.chi2_contingency([data_cont.iloc[3][0:6].values, data_cont.iloc[4][0:6].values])
print(h13)
print("\n 2021 & 2022.1 #############\n")

h14 = stats.chi2_contingency([data_cont.iloc[3][0:6].values, data_cont.iloc[5][0:6].values])
print(h14)
print("\n 2021 & 2022.2 #############\n")

h15 = stats.chi2_contingency([data_cont.iloc[4][0:6].values, data_cont.iloc[5][0:6].values])
print(h15)
print("\n 2022.1 & 2022.2 #############\n")

# ------------------------------------------------------------