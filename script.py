import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
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


# |---------- plot settings ----------|
# main colors
color_bg = "#1E1B18"
color_text = "#FFFFFF"

# color palette
palette1 = sns.color_palette("Paired", len(df['Type'].unique()))[::-1]
gradient1 = sns.color_palette("crest", len(df['year_quarter'].unique()))[::-1]

# save color palettes for reference
# define dir
colors_dir = "colors"
os.makedirs(colors_dir, exist_ok=True)

# palette 1
plt.figure()
sns.palplot(palette1)
plt.gca().set_title('Color Palette 1')
plt.tight_layout()
palette_path = os.path.join(colors_dir, "palette1.png")
plt.savefig(palette_path)
plt.close()

# gradient 1
plt.figure()
sns.palplot(gradient1)
plt.gca().set_title('Color Gradient 1')
plt.tight_layout()
palette_path = os.path.join(colors_dir, "gradient1.png")
plt.savefig(palette_path)
plt.close()

# general
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams["figure.figsize"] = 16,8

# title
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.titleweight'] = 'bold'

# colors
mpl.rcParams["figure.facecolor"] = color_bg
mpl.rcParams["axes.facecolor"] = color_bg
mpl.rcParams["savefig.facecolor"] = color_bg

# text colors
mpl.rcParams['text.color'] = color_text
mpl.rcParams['axes.labelcolor'] = color_text
mpl.rcParams['xtick.color'] = color_text
mpl.rcParams['ytick.color'] = color_text

# axis colors
mpl.rcParams['axes.edgecolor'] = color_text

# labels
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

# legends
mpl.rcParams['legend.title_fontsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['legend.frameon'] = False

# grids
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# spacing
mpl.rcParams['axes.titlepad'] = 24
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['ytick.major.pad'] = 10
mpl.rcParams['xtick.major.width'] = 0
mpl.rcParams['xtick.minor.width'] = 0
mpl.rcParams['ytick.major.width'] = 0
mpl.rcParams['ytick.minor.width'] = 0

# |------------------------------------------|
# |---------- TIME SERIES ANALYSIS ----------|
# |------------------------------------------|

# plot line chart for each type of goods
plt.figure()

ax = sns.lineplot(
    data=df,
    x='year_quarter',
    y='HFCE',
    hue='Type',
    palette=palette1,
    linewidth=0.9,
    alpha=0.9,
)

# highlight line for education
ax.plot(
    df_educ['year_quarter'],
    df_educ['HFCE'],
    color=palette1[9],
    linewidth=3,
    label='Education',
    zorder=14,
)

# glow effect
for i in range(12, 0, -1):
    ax.plot(
        df_educ['year_quarter'],
        df_educ['HFCE'],
        color=palette1[9],
        linewidth=i,
        alpha=0.05,
        zorder=13,
    )

# labels
plt.xlabel('Year')
plt.ylabel('HFCE (in million pesos)')
plt.title('Philippines Quarterly HFCE on Different Types of Goods (2000-2025)')

# add grid lines
plt.grid(alpha=0.2)

plt.tight_layout()

# save to plot dir
save_path = os.path.join(save_dir, "tsa.png")
plt.savefig(save_path)
plt.close()


# |------------------------------------------|
# |---------- CORRELATION ANALYSIS ----------|
# |------------------------------------------|

# check distribution
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
sns.histplot(df_educ['HFCE'], kde=True, bins=20, color='#ADE8F4')
plt.title('HFCE in Education', fontsize=16)
plt.xlabel("HFCE in Education (in million pesos)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=11)

plt.subplot(2, 2, 2)
sns.histplot(df_educ['GVA'], kde=True, bins=20, color='#ADE8F4')
plt.title('GVA', fontsize=16)
plt.xlabel("GVA (in million pesos)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=11)

# save to plot dir
save_path = os.path.join(save_dir, "dist.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

# apply log transform
df_educ['HFCE_log'] = np.log(df_educ['HFCE'])
df_educ['GVA_log'] = np.log(df_educ['GVA'])

# post-log
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
sns.histplot(df_educ['HFCE_log'], kde=True, bins=20, color='#DAFFBA')
plt.title('HFCE in Education (post-log)', fontsize=16)
plt.xlabel("HFCE in Education (in million pesos)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=11)

plt.subplot(2, 2, 2)
sns.histplot(df_educ['GVA_log'], kde=True, bins=20, color='#DAFFBA')
plt.title('GVA (post-log)', fontsize=16)
plt.xlabel("GVA (in million pesos)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=11)

# save to plot dir
save_path = os.path.join(save_dir, "postlog_dist.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

# scatter plot
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
sns.scatterplot(
    data=df_educ,
    x='HFCE',
    y='GVA',
    alpha=0.7,
    color=palette1[11],
)
plt.xlabel('HFCE (in million pesos)', fontsize=12)
plt.ylabel('GVA (in million pesos)', fontsize=12)
plt.title('HFCE vs GVA', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=11)

# post-log
plt.subplot(2, 2, 2)
sns.scatterplot(
    data=df_educ,
    x='HFCE_log',
    y='GVA_log',
    alpha=0.7,
    color=palette1[9],
)
plt.xlabel('log(HFCE)', fontsize=12)
plt.ylabel('log(GVA)', fontsize=12)
plt.title('HFCE vs GVA (post-log)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=11)

# save to plot dir
save_path = os.path.join(save_dir, "scatter.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()


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

# helper func
def get_event_date(start, end, event) -> float:
    return event.year + (event - start) / (end - start)

# major crisis events
yolanda = get_event_date(date(2013, 1, 1), date(2014, 1, 1), date(2013, 11, 3))
covid = get_event_date(date(2020, 1, 1), date(2021, 1, 1), date(2020, 1, 30))

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
plt.figure()

# gradient line for HFCE
for i in range(len(df_educ['year_quarter']) - 1):
    sns.lineplot(
        x=df_educ['year_quarter'][i:i+2],
        y=df_educ['HFCE'][i:i+2],
        color=gradient1[i],
        linewidth=2,
    )

# regression line
sns.lineplot(
    x=df_educ['year_quarter'],
    y=df_educ['predicted'],
    color=palette1[11],
    zorder=3, 
    linewidth=2,
)

# glow effect
for i in range(12, 0, -1):
    plt.plot(
        df_educ['year_quarter'],
        df_educ['predicted'],
        color=palette1[11],
        linewidth=i,
        alpha=0.05,
        zorder=2,
    )

# intervention lines
ax = plt.gca()

plt.axvline(x=yolanda, color=color_text, linestyle=':', label='Yolanda (2013)')
ax.text(yolanda + 0.1, ax.get_ylim()[1]*0.9, 'Yolanda (2013)', color=color_text, fontsize=14)

plt.axvline(x=covid, color=color_text, linestyle='--', label='COVID-19 (2020)')
ax.text(covid + 0.1, ax.get_ylim()[1]*0.9, 'COVID-19 (2020)', color=color_text, fontsize=14)

plt.xlabel('Year')
plt.ylabel('HFCE in education (in million pesos)')
plt.title('Philippines HFCE in Education (2000-2025)')

plt.grid(alpha=0.2)
plt.tight_layout()

# save to plot dir
save_path = os.path.join(save_dir, "itsa.png")
plt.savefig(save_path)
plt.close()