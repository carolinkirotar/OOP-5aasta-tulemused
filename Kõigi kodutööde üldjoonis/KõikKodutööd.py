import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy as rp
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols
import matplotlib.gridspec as gridspec


# Dataframe 

df = pd.read_csv('/Users/carolin/Documents/Magister/AndmedRühmagaVALMIS.csv', decimal=',')
df_copy = df

df_copy = df_copy.replace('-', '0', regex=True)

cols = df.columns
df_copy[cols[1:]] = df_copy[cols[1:]].apply(pd.to_numeric, errors='coerce')

df_copy['aasta'] = df_copy['aasta'].astype(str)

df_copy.rename(columns={'aasta': 'Õppeaasta', '1.nädalaÜlesanded': '1. nädal', '2.nädalaKoduülesanded': '2. nädal',
                        '3.nädalaKoduülesanded': '3. nädal', '4.nädalaKoduülesanded':'4. nädal', '5.nädalaKoduülesanded':'5. nädal',
                        '6.nädalaKoduülesanded': '6. nädal', '7.nädalaKoduülesanded': '7. nädal', '8.nädalaKoduülesanded': '8. nädal',
                        '9.nädalaKoduülesanded': '9. nädal', '10.nädalaKoduülesanded': '10. nädal', '11.nädalaKoduülesanded':'11. nädal',
                        '12.nädalaKoduülesanded':'12. nädal'}, inplace=True)

focused_row = 'KodutöödeLõpptulemus'

# -------------------------------------------------------------

sns.set_style("whitegrid", {'grid.color': '.8'})

fig3 = plt.figure(constrained_layout=True, figsize=(12, 16))
gs = gridspec.GridSpec(ncols=3, nrows=4, figure=fig3, wspace=0.5, hspace=0.5)
f3_ax1 = fig3.add_subplot(gs[0, 0])
f3_ax2 = fig3.add_subplot(gs[0, 1])
f3_ax3 = fig3.add_subplot(gs[0, 2])
f3_ax4 = fig3.add_subplot(gs[1, 0])
f3_ax5 = fig3.add_subplot(gs[1, 1])
f3_ax6 = fig3.add_subplot(gs[1, 2])
f3_ax7 = fig3.add_subplot(gs[2, 0])
f3_ax8 = fig3.add_subplot(gs[2, 1])
f3_ax9 = fig3.add_subplot(gs[2, 2])
f3_ax10 = fig3.add_subplot(gs[3, 0])
f3_ax11 = fig3.add_subplot(gs[3, 1])
f3_ax12 = fig3.add_subplot(gs[3, 2])


sns.boxplot(data=df_copy, x="Õppeaasta", y="1. nädal", linewidth=1, color='white', ax = f3_ax1, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="2. nädal", linewidth=1, color='white', ax = f3_ax2, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="3. nädal", linewidth=1, color = 'white', ax = f3_ax3, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="4. nädal", linewidth=1, color = 'white', ax = f3_ax4, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="5. nädal", linewidth=1, color = 'white', ax = f3_ax5, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="6. nädal", linewidth=1, color = 'white', ax = f3_ax6, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="7. nädal", linewidth=1, color = 'white', ax = f3_ax7, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="8. nädal", linewidth=1, color = 'white', ax = f3_ax8, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="9. nädal", linewidth=1, color='white', ax = f3_ax9, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="10. nädal", linewidth=1, color='white', ax = f3_ax10, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="11. nädal", linewidth=1, color='white', ax = f3_ax11, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="12. nädal", linewidth=1, color='white', ax = f3_ax12, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

fig3.tight_layout(pad=0.2)
fig3.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.05) 

# -----------------------------------------------------------

# Mann-Whitney U-test
first_year = df_copy[df_copy['Õppeaasta'] == '2018']
second_year = df_copy[df_copy['Õppeaasta'] == '2019']
third_year = df_copy[df_copy['Õppeaasta'] == '2020']
fourth_year = df_copy[df_copy['Õppeaasta'] == '2021']
fifth_year1 = df_copy[df_copy['Õppeaasta'] == '2022.1']
fifth_year2 = df_copy[df_copy['Õppeaasta'] == '2022.2']

between18and19 = stats.mannwhitneyu(x=first_year[focused_row], y=second_year[focused_row], alternative = 'two-sided')
between18and20 = stats.mannwhitneyu(x=first_year[focused_row], y=third_year[focused_row], alternative = 'two-sided')
between18and21 = stats.mannwhitneyu(x=first_year[focused_row], y=fourth_year[focused_row], alternative = 'two-sided')
between18and221 = stats.mannwhitneyu(x=first_year[focused_row], y=fifth_year1[focused_row], alternative = 'two-sided')
between18and222 = stats.mannwhitneyu(x=first_year[focused_row], y=fifth_year2[focused_row], alternative = 'two-sided')
between19and20 = stats.mannwhitneyu(x=second_year[focused_row], y=third_year[focused_row], alternative = 'two-sided')
between19and21 = stats.mannwhitneyu(x=second_year[focused_row], y=fourth_year[focused_row], alternative = 'two-sided')
between19and221 = stats.mannwhitneyu(x=second_year[focused_row], y=fifth_year1[focused_row], alternative = 'two-sided')
between19and222 = stats.mannwhitneyu(x=second_year[focused_row], y=fifth_year2[focused_row], alternative = 'two-sided')
between20and21 = stats.mannwhitneyu(x=third_year[focused_row], y=fourth_year[focused_row], alternative = 'two-sided')
between20and221 = stats.mannwhitneyu(x=third_year[focused_row], y=fifth_year1[focused_row], alternative = 'two-sided')
between20and222 = stats.mannwhitneyu(x=third_year[focused_row], y=fifth_year2[focused_row], alternative = 'two-sided')
between21and221 = stats.mannwhitneyu(x=fourth_year[focused_row], y=fifth_year1[focused_row], alternative = 'two-sided')
between21and222 = stats.mannwhitneyu(x=fourth_year[focused_row], y=fifth_year2[focused_row], alternative = 'two-sided')
between221and222 = stats.mannwhitneyu(x=fifth_year1[focused_row], y=fifth_year2[focused_row], alternative = 'two-sided')

print("2018 ja 2019: " + str(between18and19))
print("2018 ja 2020: " + str(between18and20))
print("2018 ja 2021: " + str(between18and21))
print("2018 ja 2022.1: " + str(between18and221))
print("2018 ja 2022.2: " + str(between18and222))
print("2019 ja 2020: " + str(between19and20))
print("2019 ja 2021: " + str(between19and21))
print("2019 ja 2022.1: " + str(between19and221))
print("2019 ja 2022.2: " + str(between19and222))
print("2020 ja 2021: " + str(between20and21))
print("2020 ja 2022.1: " + str(between20and221))
print("2020 ja 2022.2: " + str(between20and222))
print("2021 ja 2022.1: " + str(between21and221))
print("2021 ja 2022.2: " + str(between21and222))
print("2022.1 ja 2022.2: " + str(between221and222))

# --------------------------------------------------------------------------

# Kruskal-Wallis test

tulemus = stats.kruskal(df_copy[focused_row][df_copy['Õppeaasta'] == '2018'],
               df_copy[focused_row][df_copy['Õppeaasta'] == '2019'],
               df_copy[focused_row][df_copy['Õppeaasta'] == '2020'],
               df_copy[focused_row][df_copy['Õppeaasta'] == '2021'],
               df_copy[focused_row][df_copy['Õppeaasta'] == '2022.1'],
               df_copy[focused_row][df_copy['Õppeaasta'] == '2022.2'])
print("\n")
print("Kruskali test:" + str(tulemus))
print(round(tulemus.pvalue, 4))

# --------------------------------------------------------------------------

plt.show()




