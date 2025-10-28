import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from datetime import date

# define plot dir
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# load the dataframe
df = pd.read_csv("data.csv")

# |---------- preprocessing ----------|

# simplify the column names of HFCE and GVA
df = df.rename(columns={'HFCE (in million pesos)': 'HFCE'})
df = df.rename(columns={'GVA (in million pesos)': 'GVA'})

# drop rows with NA fields
df = df.dropna()

# dict to convert quarter data to 
# numerical format (for plotting)
quarters_dict = {
    'Q1': 0.25, 
    'Q2': 0.50, 
    'Q3': 0.75, 
    'Q4': 1.00, 
}

# add a column to store the numerical
# representation of each yearly quarter
df['year_quarter'] = df['Year'] + df['Quarter'].map(quarters_dict)

# education HFCE
df_educ = df[df['Type'] == 'Education'].copy()


# |------------------------------------|
# |----------- NUTSHELL PLOT ----------|
# |------------------------------------|

# helper func
def get_event_date(start, end, event) -> float:
    return event.year + (event - start) / (end - start)

# major crisis events
yolanda = get_event_date(date(2013, 1, 1), date(2014, 1, 1), date(2013, 11, 3))
covid = get_event_date(date(2020, 1, 1), date(2021, 1, 1), date(2020, 1, 30))

# Dual axis time series chart
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(
    df_educ['year_quarter'], 
    df_educ['HFCE'], 
    label=df_educ['Type'].unique(), 
    color='#1f77b4'
)

ax1.set_xlabel('Year')
ax1.set_ylabel('HFCE in education (in million pesos)')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(
    df_educ['year_quarter'], 
    df_educ['GVA'], 
    linestyle='--', 
    label='GVA',
    color='#ff7f0e',
)

ax2.set_ylabel('GVA (in million pesos)')
ax2.tick_params(axis='y')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc='center left',
    bbox_to_anchor=(1.2, 0.8),
    borderaxespad=0,
    fontsize='small'
)

# intervention lines
ax1.axvline(x=yolanda, color='#2ca02c', linestyle='--', label='Yolanda (2013)')
ax1.axvline(x=covid, color='#d62728', linestyle='--', label='COVID-19 (2020)')

ax1.grid(True)

plt.title('Philippines Quarterly HFCE in Education and GVA from 2000 to 2025')
plt.tight_layout()

# save to plot dir
save_path = os.path.join(save_dir, "nutshell.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')


# |------------------------------------------|
# |---------- TIME SERIES ANALYSIS ----------|
# |------------------------------------------|

plt.figure(figsize=(12, 6))

# plot line chart for each type of goods
for category, group in df.groupby('Type'):
    plt.plot(group['year_quarter'], group['HFCE'], label=category)

plt.xlabel('Year')
plt.ylabel('HFCE (in million pesos)')
plt.title('Philippines Quarterly HFCE from 2000 to 2025')

plt.legend(
    loc='center left',  
    bbox_to_anchor=(1.02, 0.8), 
    borderaxespad=0, 
    fontsize='small', 
)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.grid(True)

# save to plot dir
save_path = os.path.join(save_dir, "tsa.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')


# |------------------------------------------|
# |---------- CORRELATION ANALYSIS ----------|
# |------------------------------------------|

# check distribution
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
sns.histplot(df_educ['HFCE'], kde=True, bins=20, color='skyblue', edgecolor='k')
plt.title('HFCE in Education')

plt.subplot(2, 2, 2)
sns.histplot(df_educ['GVA'], kde=True, bins=20, color='skyblue', edgecolor='k')
plt.title('GVA')

# save to plot dir
save_path = os.path.join(save_dir, "dist.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# apply log transform
df_educ['HFCE_log'] = np.log(df_educ['HFCE'])
df_educ['GVA_log'] = np.log(df_educ['GVA'])

# post-log
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
sns.histplot(df_educ['HFCE_log'], kde=True, bins=20, color='lightgreen', edgecolor='k')
plt.title('HFCE in Education (post-log)')

plt.subplot(2, 2, 2)
sns.histplot(df_educ['GVA_log'], kde=True, bins=20, color='lightgreen', edgecolor='k')
plt.title('GVA (post-log)')

# save to plot dir
save_path = os.path.join(save_dir, "postlog_dist.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# scatter plot
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
plt.scatter(df_educ['HFCE'], df_educ['GVA'], alpha=0.7)
plt.xlabel('HFCE (in million pesos)')
plt.ylabel('GVA (in million pesos)')
plt.title('HFCE vs GVA')

# post-log
plt.subplot(2, 2, 2)
plt.scatter(df_educ['HFCE_log'], df_educ['GVA_log'], alpha=0.7)
plt.xlabel('log(HFCE)')
plt.ylabel('log(GVA)')
plt.title('HFCE vs GVA (post-log)')

# save to plot dir
save_path = os.path.join(save_dir, "scatter.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')



# Pearson's test
corr,_ = pearsonr(
    df_educ['HFCE'], 
    df_educ['GVA'], 
)

print('Pearson\'s correlation: %.4f' % corr)

corr_log,_ = pearsonr(
    df_educ['HFCE_log'], 
    df_educ['GVA_log'], 
)

print('Pearson\'s correlation (post-log): %.4f' % corr_log)


# |-------------------------------------------------------|
# |----------- INTERRUPTED TIME SERIES ANALYSIS ----------|
# |-------------------------------------------------------|

# intervention
df_educ['time'] = df_educ['year_quarter'] - df_educ['year_quarter'].min()

df_educ['yolanda'] = (df_educ['year_quarter'] >= yolanda).astype(int)
df_educ['time_after_yolanda'] = np.where(df_educ['year_quarter'] >= yolanda,
                                     df_educ['year_quarter'] - yolanda, 0)

df_educ['covid'] = (df_educ['year_quarter'] >= covid).astype(int)
df_educ['time_after_covid'] = np.where(df_educ['year_quarter'] >= covid,
                                   df_educ['year_quarter'] - covid, 0)

# fit model
X = df_educ[['time', 'yolanda', 'time_after_yolanda', 'covid', 'time_after_covid']]
X = sm.add_constant(X)
model = sm.OLS(df_educ['HFCE'], X).fit()

print(model.summary())

# predict values
df_educ['predicted'] = model.predict(X)

# plot results
plt.figure(figsize=(12, 6))

plt.plot(df_educ['year_quarter'], df_educ['HFCE'], label='Observed', alpha=0.7, color='darkblue')
plt.plot(df_educ['year_quarter'], df_educ['predicted'], '-r', label='Predicted (ITS model)', linewidth=2)

# intervention lines
plt.axvline(x=yolanda, color='#2ca02c', linestyle='--', label='Yolanda (2013)')
plt.axvline(x=covid, color='#d62728', linestyle='--', label='COVID-19 (2020)')

plt.xlabel('Year')
plt.ylabel('HFCE in education (in million pesos)')
plt.title('Education HFCE from 2000 to 2025 with Yolanda and COVID-19 Interventions')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), fontsize='small')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 1])

# save to plot dir
save_path = os.path.join(save_dir, "itsa.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')