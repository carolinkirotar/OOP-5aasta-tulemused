import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import researchpy as rp
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols


# Dataframe 

df = pd.read_csv('/Users/carolin/Documents/Magister/AndmedRühmaga2.csv', decimal=',')
df_copy = df

df_copy = df_copy.replace('-', '0', regex=True)

cols = df.columns
df_copy[cols[1:]] = df_copy[cols[1:]].apply(pd.to_numeric, errors='coerce')

df_copy['aasta'] = df_copy['aasta'].astype(str)

focused_row = "Eksamitöö"


# ----------------------------------------------------

from scipy.stats import kstest, norm
my_data = norm.rvs(size=1000)
ks_statistic, p_value = kstest(df_copy[focused_row], 'norm')
print(ks_statistic, p_value)

# --------------------------------------------------

hist = df_copy[focused_row].hist()
plt.show()