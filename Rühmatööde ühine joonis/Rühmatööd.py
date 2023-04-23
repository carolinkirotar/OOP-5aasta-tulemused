import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy as rp
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols


# Dataframe 

df = pd.read_csv('/Users/carolin/Documents/Magister/AndmedRühmagaVALMIS.csv', decimal=',')
df_copy = df

df_copy = df_copy.replace('-', '0', regex=True)

cols = df.columns
df_copy[cols[1:]] = df_copy[cols[1:]].apply(pd.to_numeric, errors='coerce')

df_copy['aasta'] = df_copy['aasta'].astype(str)

df_copy.rename(columns={'aasta': 'Õppeaasta', 'RühmatööLõpptulemus': 'Rühmatöö lõpptulemus',
                        'RühmatööNr.1': 'Rühmatöö 1', 'RühmatööNr.2':'Rühmatöö 2', 'RühmatööEsitlus':'Rühmatöö esitlus'}, inplace=True)

focused_row = 'Rühmatöö lõpptulemus'

# -------------------------------------------------------------

sns.set_style("whitegrid", {'grid.color': '.8'})

fig3 = plt.figure(constrained_layout=True, figsize=(14, 8))
gs = fig3.add_gridspec(3, 4)
f3_ax1 = fig3.add_subplot(gs[:, :-1])
f3_ax2 = fig3.add_subplot(gs[0, -1])
f3_ax3 = fig3.add_subplot(gs[1, -1])
f3_ax4 = fig3.add_subplot(gs[2, -1])

ax = sns.violinplot(data=df_copy, x="Õppeaasta", y=focused_row, inner="quart", linewidth=1, scale="count", palette='magma', ax = f3_ax1)
plt.setp(ax.collections, alpha=.4)

box_plot = sns.boxplot(data=df_copy, x="Õppeaasta", y=focused_row, ax = f3_ax1,
                       boxprops={'zorder': 2, 'fill': None}, width=0.3, showmeans=True,
                       meanprops={"marker":"o", "markerfacecolor":"darkred",
                                  "markeredgecolor":"black","markersize":"15"})
f3_ax1.set_ylim(-0.5, 14)

sns.boxplot(data=df_copy, x="Õppeaasta", y="Rühmatöö 1", linewidth=1, color='white', ax = f3_ax2, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="Rühmatöö 2", linewidth=1, color = 'white', ax = f3_ax3, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})

sns.boxplot(data=df_copy, x="Õppeaasta", y="Rühmatöö esitlus", linewidth=1, color = 'white', ax = f3_ax4, width=0.4, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"darkred", "markeredgecolor":"black","markersize":"9"})
plt.ylim(-0.15, 3.2)

fig3.tight_layout()
fig3.subplots_adjust(bottom=0.1, right=0.95, left=0.05) 

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

# näidata keskmist ja sd keskel
summary = rp.summary_cont(df_copy[focused_row].groupby(df_copy['Õppeaasta']))
means = np.array(df_copy.groupby(['Õppeaasta'])[focused_row].mean())
stds = np.array(df_copy.groupby(['Õppeaasta'])[focused_row].std())
conf = np.array(summary['95% Conf.'])
interval = np.array(summary['Interval'])

for i in range(len(means)):
    text = ' μ={:.2f}\n σ={:.2f}\n 95% Conf=\n({:.2f}, {:.2f})'.format(means[i], stds[i], conf[i], interval[i])
    box_plot.annotate(text, xy=(i, means[i]), xytext=(i+0.04, means[i]-1.5), ha='left', fontsize='8',
                      bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=1'))


# ---------------------------------------------------------------

# näidata keskmise ühendusjooni
mean = df_copy.groupby(['Õppeaasta'])[focused_row].mean()
box_plot.plot(mean.values, 'r-o', linewidth=3, alpha=0.5)

# ---------------------------------------------------------------

# näidata iga aasta juures n arvu
nobs = df_copy['Õppeaasta'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["(n: " + i + ")" for i in nobs]

pos = range(len(nobs))
for tick, label in zip(pos, box_plot.get_xticklabels()):
   box_plot.text(pos[tick], -1.28, nobs[tick],
            ha='center',
            size='small')

plt.show()

