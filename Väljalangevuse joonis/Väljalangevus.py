import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Dataframe 

df = pd.read_csv('/Users/carolin/Documents/Magister/AndmedRühmagaVALMIS.csv', decimal=',')
df_copy = df

cols = df.columns
df_copy[cols[1:30]] = df_copy[cols[1:30]].apply(pd.to_numeric, errors='coerce')

df_copy['aasta'] = df_copy['aasta'].astype(str)

df_copy['Väljalangemine'] = df_copy['Väljalangemine'].astype(str)

focused_row = 'Väljalangemine'

# -------------------------------------------------------------

sns.set_style("whitegrid")

plt.subplots(figsize=(8, 18))

pal = sns.color_palette("magma")
pal.as_hex()
print(pal.as_hex())

palette = {"Üldse pole teinud":"#221150",
           "Esimestel nädalatel":"mediumorchid", 
           "Enne esimest kontrolltööd":"hotpink",
           "Enne teist kontrolltööd":"#f8765c",
           "Semestri lõpus":"#febb81"}

# ['#6e90bf', '#aab8d0', '#e4e5eb', '#f2dfdd', '#d9a6a4', '#c26f6d']
# ['#221150', '#5f187f', '#982d80', '#d3436e', '#f8765c', '#febb81']

count3 = df_copy.groupby(['aasta'])['Väljalangemine'].size().reset_index(name='count')
#print(count3)
print("##########")

perc = df_copy.groupby(['aasta'])['Väljalangemine'].value_counts(normalize=True, sort=False).mul(100).reset_index(name='perc')
print(perc)
print("##########\n")

hue_order = ['Üldse pole teinud', 'Esimestel nädalatel', 'Enne esimest kontrolltööd', 'Enne teist kontrolltööd', 'Semestri lõpus']
ax = sns.lineplot(data=perc, x="aasta", y='perc', hue='Väljalangemine', hue_order=hue_order, palette=palette, marker="o")

# -------------------------------------------------------

data_cont = pd.crosstab(df_copy['aasta'], df_copy['Väljalangemine'])
print(data_cont)
print("\n")

t = stats.chi2_contingency(data_cont)
print(t)
print("\n#############\n")

plt.ylabel("Väljalangenute % kursuse alustajatest", fontweight='bold')
plt.xlabel('Õppeaasta', fontweight='bold', labelpad=25)
plt.show()


