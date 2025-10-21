#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: 02_data_cleaning.ipynb
Conversion Date: 2025-10-21T10:49:36.473Z
"""

# # Fighters Analysis


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("dark")

# Who are the current belt holders ?


fighters_df = pd.read_csv('data/Fighters.csv')
fighters_df.head()

champs = fighters_df[fighters_df['Belt']]

print("The current UFC champions are :")
for champ in champs['Full Name'].tolist():
    print(f'- {champ}')

# What is the most common stance between fighters ? champions ?


sns.countplot(data=fighters_df, x='Stance')
plt.show()

# Fighters are most likely to have an Orthodox Stance


# Which fighter(s) has the longest Reach ever ?


fighters_df[fighters_df['Reach'] == fighters_df['Reach'].max()]

# Height Vs Weight


sns.scatterplot(
    data=fighters_df[~(fighters_df['Wt.'] > 500)], x='Ht.', y='Wt.', hue='Reach')
plt.show()

# There is not a big correlation between the two variables


# # Events Analysis


events_df = pd.read_csv('data/Events.csv', parse_dates=['Date'])
events_df.head()

# Which year had the most events ?


events_df['Year'] = events_df['Date'].dt.year

events_per_year = events_df.groupby('Year').size()

plt.figure(figsize=(12, 6))
sns.barplot(x=events_per_year.index, y=events_per_year.values, color='C0')
plt.title("Number of UFC Events per Year")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Number of Events")
plt.show()

# - 2014 had the most events.


# Which location witnessed the most events


events_per_location = events_df['Location'].value_counts().head(20)

events_per_location.plot(kind='barh', figsize=(10, 6))
plt.title("Top 10 UFC Event Locations")
plt.xlabel("Number of Events")
plt.ylabel("Location")
plt.gca().invert_yaxis()
plt.show()

# Las Vegas is by far the location with most events


# # Fights Analysis


fights_df = pd.read_csv('data/Fights.csv')
fights_df.head()

# Fight rounds distribution


sns.countplot(x="Round", data=fights_df)
plt.show()

# What is the most common fight ending's method?


top_methods = fights_df['Method'].value_counts().head(10).index
sns.countplot(data=fights_df[fights_df['Method'].isin(top_methods)],
              x='Method', order=top_methods)
plt.xticks(rotation=90)
plt.show()

sum_of_attacks = fights_df.groupby('Weight_Class')[
    ['KD_1', 'KD_2', 'TD_1', 'TD_2', 'SUB_1', 'SUB_2', 'STR_1', 'STR_2']].sum().sum(axis=1)

fights_per_weight_class = fights_df.groupby('Weight_Class').size()

(sum_of_attacks / fights_per_weight_class).sort_values(ascending=False)

# - Women's Flyweight, Catch Weight and Featherweight Fights are probably the most enjoyable in terms of action and strikes


# - What is the average control time per weight division?


control_by_weight = fights_df.groupby('Weight_Class')[
    ['Ctrl_1', 'Ctrl_2']].mean().sum(axis=1).sort_values(ascending=False)
control_by_weight.sort_values().plot(kind='barh', color='C0', figsize=(10, 6))
plt.title('Average Control Time per Weight Class')
plt.xlabel('Average Control Time (seconds)')
plt.ylabel('Weight Class')
plt.tight_layout()
plt.show()

# HEHE ! This is almost like ranking the most boring weight divisions. Fights with higher control time are probably the ones you don't wanna watch to for entertainment.


# - Which method is dominating UFC fights?


method_counts = fights_df['Method'].value_counts()

# Keep only top counts
top_counts = method_counts.reindex(top_methods, fill_value=0)
# Sum other count
other_count = method_counts[~method_counts.index.isin(top_methods)].sum()


labels = top_methods.tolist() + (['Other'] if other_count > 0 else [])
sizes = top_counts.values.tolist() + ([other_count] if other_count > 0 else [])

fig, ax = plt.subplots(figsize=(12, 8))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.8,
    labeldistance=1.05,
    explode=[0.05]*11
    # wedgeprops = {'ls':'-','lw': 3,'ec':'black'}
)
ax.axis('equal')
ax.set_title('Fight Ending Methods (Top 10 + Other)')

plt.tight_layout()
plt.show()

print(
    f"Nearly {int(len(fights_df)*0.122)} of the UFC fights was ended by KO/TKO Punches")

# > For now, this is enough for **EDA**. This is an iterative process so I am gonna do much more of it when building the ML model.
