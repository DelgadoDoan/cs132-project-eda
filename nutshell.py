import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager as fm, pyplot as plt
from datetime import date


# font dir
os.makedirs("fonts", exist_ok=True)

urls = [
    "https://github.com/openmaptiles/fonts/raw/master/roboto/Roboto-Light.ttf",
    "https://github.com/openmaptiles/fonts/raw/master/roboto/Roboto-Regular.ttf",
    "https://github.com/openmaptiles/fonts/raw/master/roboto/Roboto-Medium.ttf",
    "https://github.com/openmaptiles/fonts/raw/master/roboto/Roboto-Bold.ttf"
]

for url in urls:
    filename = os.path.join("fonts", os.path.basename(url))
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)

font_files = [os.path.join("fonts", f) for f in os.listdir("fonts") if f.endswith(".ttf")]

for font_file in font_files:
    fm.fontManager.addfont(font_file)


# @title Colors
colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#000000", "#FFFFFF"]
colors_grad = sns.color_palette('flare_r',  12)
colors_heat1 = sns.color_palette('flare_r', as_cmap=True)
colors_heat2 = sns.diverging_palette(315, 261, s=74, l=50, center='dark', as_cmap=True)

color_bg = "#1B181C"
color_text = "#FFFFFF"


# @title Plot settings
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams["figure.figsize"] = 16,8

# Text
mpl.rcParams['font.family'] = 'Roboto'

# Title
mpl.rcParams['figure.titlesize'] = 32
mpl.rcParams['axes.titlesize'] = 32
mpl.rcParams['axes.titleweight'] = 'bold'

# Labels
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22

# Spacing
mpl.rcParams['axes.titlepad'] = 72
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['ytick.major.pad'] = 10
mpl.rcParams['xtick.major.width'] = 0
mpl.rcParams['xtick.minor.width'] = 0
mpl.rcParams['ytick.major.width'] = 0
mpl.rcParams['ytick.minor.width'] = 0

# Spines and grids
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = False

# Legends
mpl.rcParams['legend.title_fontsize'] = 18
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['legend.frameon'] = False

# Bars
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'

# Colors
mpl.rcParams["figure.facecolor"] = color_bg
mpl.rcParams["axes.facecolor"] = color_bg
mpl.rcParams["savefig.facecolor"] = color_bg

# Text colors
mpl.rcParams['text.color'] = color_text
mpl.rcParams['axes.labelcolor'] = color_text
mpl.rcParams['xtick.color'] = color_text
mpl.rcParams['ytick.color'] = color_text

# Line colors
mpl.rcParams['axes.edgecolor'] = color_text


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

# helper function
def get_event_date(start, end, event) -> float:
    return event.year + (event - start) / (end - start)

# major crisis events
yolanda = get_event_date(date(2013, 1, 1), date(2014, 1, 1), date(2013, 11, 3))
covid = get_event_date(date(2020, 1, 1), date(2021, 1, 1), date(2020, 1, 30))

# plot
# fig, ax1 = plt.subplots()

# # HFCE line
# sns.lineplot(
#     data=df_educ, x='year_quarter', y='HFCE',
#     ax=ax1, color=colors[1], linewidth=2, 
#     label='HFCE (Education)', legend=False,
# )

# # GVA line
# ax2 = ax1.twinx()
# sns.lineplot(
#     data=df_educ, x='year_quarter', y='GVA',
#     ax=ax2, color=colors[2], linestyle='--', linewidth=2, 
#     label='GVA', legend=False,
# )

# # axes labels
# ax1.set_xlabel('Year')
# ax1.set_ylabel('HFCE in Education (million pesos)')
# ax2.set_ylabel('GVA (million pesos)')

# # events
# ax1.axvline(x=yolanda, color=color_text, linestyle=':', alpha=0.6)
# ax1.text(yolanda + 0.1, ax1.get_ylim()[1]*0.9, 'Yolanda (2013)', color=color_text, fontsize=16)

# ax1.axvline(x=covid, color=color_text, linestyle='--', alpha=0.6)
# ax1.text(covid + 0.1, ax1.get_ylim()[1]*0.9, 'COVID-19 (2020)', color=color_text, fontsize=16)

# # plot title
# plt.title('Philippines Quarterly HFCE in Education and GVA (2000–2025)')
# ax1.grid(alpha=0.2)

# # legends
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(
#     lines_1 + lines_2, labels_1 + labels_2,
# )

# plt.tight_layout()

# # save to plot dir
# save_path = os.path.join(save_dir, "nutshell.png")
# plt.savefig(save_path)

plt.figure()

# HFCE
sns.lineplot(
    x=df_educ['year_quarter'],
    y=df_educ['HFCE'],
    label='HFCE in education',
    color=colors[1],
    linewidth=2,
)

# GVA
sns.lineplot(
    x=df_educ['year_quarter'],
    y=df_educ['GVA'],
    label='GVA',
    color=colors[2],
    linewidth=2,
    linestyle='--',
)

# intervention lines
ax = plt.gca()
plt.axvline(x=yolanda, color=color_text, linestyle=':', label='Yolanda (2013)')
ax.text(yolanda + 0.1, ax.get_ylim()[1]*0.9, 'Yolanda (2013)', color=color_text, fontsize=14)

plt.axvline(x=covid, color=color_text, linestyle='--', label='COVID-19 (2020)')
ax.text(covid + 0.1, ax.get_ylim()[1]*0.9, 'COVID-19 (2020)', color=color_text, fontsize=14)

# labels
plt.xlabel('Year')
plt.ylabel('Amount in million pesos')
plt.title('Philippines Quarterly HFCE in Education and GVA (2000–2025)')

# plot grid
plt.grid(alpha=0.2)
plt.tight_layout()

# save to plot dir
save_path = os.path.join(save_dir, "nutshell.png")
plt.savefig(save_path)
plt.close()