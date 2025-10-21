#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: 01_web_scraping.ipynb
Conversion Date: 2025-10-21T10:49:03.171Z
"""

# # Data Cleaning


# ### Cleaning Fighters Data


import pandas as pd
import numpy as np

fighters_df = pd.read_csv('raw_data/raw_fighters.csv')

fighters_df.head()


def check_types(df):
    print(df.dtypes)


fighters_df.columns

check_types(fighters_df)


def number_of_values(df):
    print("================ Number of unique values for each col ================")
    for col in df.columns.tolist():
        print(f'{col} : {df[col].nunique()}')
    print("="*70)


number_of_values(fighters_df)

# print(len("================ Columns having missing values ================"))


def print_missing_values(df):
    print("================ Columns having missing values ================")
    for col in df.columns.tolist():
        if df[col].isna().sum():
            print(f'{col} : {df[col].isna().sum()}')
    print("="*64)


# I did this because the Full Name column would be empty in case one of the columns is empty
fighters_df.fillna({'First': ''}, inplace=True)
fighters_df.fillna({'Last': ''}, inplace=True)

fighters_df['Full Name'] = fighters_df['First'] + ' ' + fighters_df['Last']
fighters_df.drop(columns=['First', 'Last'], inplace=True)
cols = fighters_df.columns.tolist()
cols = [cols.pop()] + cols
fighters_df = fighters_df[cols]

# Height, Weight ,Reach and Stance are missing. I think I can use imputation as follows :
# - Height, Weight and Reach are imputed by the median weight of fighter's current weight_class
# - For Stance I'm gonna simply use mode imputation
#


fighters_df.fillna({"First": "Uknown"}, inplace=True)
fighters_df.fillna({"Nickname": "No Nickname"}, inplace=True)
fighters_df.fillna(
    {"Stance": fighters_df['Stance'].mode().iloc[-1]}, inplace=True)


# Ht. column
def format_height(height=""):
    if height == '--':
        return np.nan
    height = height.rstrip('"')
    height = height.replace("' ", ".")
    return height


fighters_df['Ht.'] = fighters_df['Ht.'].apply(format_height)
fighters_df['Ht.'] = fighters_df['Ht.'].astype('float32')

# Wt. column


def format_weight(height=""):
    if height == '--':
        return np.nan
    height = height.rstrip(' lbs.')
    return height


fighters_df['Wt.'] = fighters_df['Wt.'].apply(format_weight)
fighters_df['Wt.'] = fighters_df['Wt.'].astype('float32')

# Reach column


def format_reach(height=""):
    if height == '--':
        return np.nan
    height = height.rstrip('"')
    return height


fighters_df['Reach'] = fighters_df['Reach'].apply(format_reach)
fighters_df['Reach'] = fighters_df['Reach'].astype('float32')

# Making stance column categorical
fighters_df['Stance'] = fighters_df['Stance'].astype('category')

# W - L - D can not be so big.
# It is better to make them only 32-bits
for col in ['W', 'L', 'D']:
    fighters_df[col] = fighters_df[col].astype('int32')

fighters_df.head()

print_missing_values(fighters_df)

# I think this is enough for cleaning fighters data


fighters_df.to_csv('data/Fighters.csv', index=False)

# ### CLeaning Events Data
#


events_df = pd.read_csv('raw_data/raw_events.csv')
events_df.head()

check_types(events_df)

print_missing_values(events_df)

# ! Perfect


events_df['Date'] = pd.to_datetime(events_df['Date'], format="%B %d, %Y")
events_df.head()

events_df.to_csv('data/Events.csv', index=False)

# Now let's get to the largest dataframe :)


# ### Cleaning Fights Data


fights_df = pd.read_csv(
    'raw_data/raw_fights_detailed.csv').set_index('Fight_Id')

check_types(fights_df)


def values_count(df):
    for col in df.columns.tolist():
        yield col, df[col].value_counts(dropna=False)

# usage:
# gen = unique_values(fights_df)
# next(gen)  # returns (column_name, value_counts_series) for the next column


cols = values_count(fights_df)

# I am going to do this for the rest of columns


def clean_missing_fight_detail(col):
    fights_df[col] = (fights_df[col]
                      .replace('--', pd.NA)
                      .astype('Int32'))


def make_categorical(col):
    fights_df[col] = fights_df[col].astype('category')


def calculate_pct(pct):
    if pd.isna(pct) or pct == '---':
        return pd.NA
    pct = pct.rstrip('%')
    return int(pct)/100


def parse_seconds_from_time(time):
    if pd.isna(time) or time == '--':
        return pd.NA
    minutes, seconds = time.split(':')
    return int(minutes) * 60 + int(seconds)


def of_to_pct(exp):
    if pd.isna(exp):
        return pd.NA
    if exp == '0 of 0':
        return 0

    x, y = exp.split('of')
    x, y = int(x), int(y)
    try:
        return round(x/y, 2)
    except ZeroDivisionError:
        return 0


def remove_quotation(s):
    if s.startswith("'"):
        return s.strip("'")
    else:
        return s.strip('"')


def clean_fights_df():
    # Convert these columns to int and replace missing values '--' with NA
    clean_missing_fight_detail('KD_1')
    clean_missing_fight_detail('KD_2')
    clean_missing_fight_detail('STR_1')
    clean_missing_fight_detail('STR_2')
    clean_missing_fight_detail('TD_1')
    clean_missing_fight_detail('TD_2')
    clean_missing_fight_detail('SUB_1')
    clean_missing_fight_detail('SUB_2')

    # Converting categorical columns
    make_categorical('Weight_Class')
    make_categorical('Result_1')
    make_categorical('Result_2')

    fights_df['Round'] = fights_df['Round'].astype('int8')

    # Event_Id is saved twice due to table joins
    fights_df.rename(columns={'Event_Id_x': 'Event_Id'}, inplace=True)
    # Removing redundant columns
    redundant_cols = ['Kd_1', 'Kd_2', 'Td_1',
                      'Td_2', 'Win/No Contest/Draw', 'Event_Id_y']
    fights_df.drop(columns=redundant_cols, inplace=True)

    # Values are transformed from string 'X%' to float (X/100)
    pct_cols = ['Sig. Str. %', 'Td %']
    for col in pct_cols:
        fights_df[f'{col}_1'] = fights_df[f'{col}_1'].apply(calculate_pct)
        fights_df[f'{col}_2'] = fights_df[f'{col}_2'].apply(calculate_pct)
    # Cols that are in this form : x of y
    of_cols = ['Head_', 'Body_', 'Leg_', 'Distance_',
               'Clinch_', 'Ground_', 'Total Str._', 'Sig. Str._']
    # This data is supposed to be already scraped
    # But I did not scrape it to do it myself (faster)
    for col in of_cols:
        fights_df[f"{col}%_1"] = fights_df[f"{col}1"].apply(of_to_pct)
        fights_df[f"{col}%_2"] = fights_df[f"{col}2"].apply(of_to_pct)
        fights_df.drop(columns=[f"{col}1", f"{col}2"], inplace=True)

    # Stripping quotation marks that came with scraped data
    fights_df['Weight_Class'] = fights_df['Weight_Class'].apply(
        remove_quotation)
    fights_df['Method'] = fights_df['Method'].apply(remove_quotation)
    fights_df['Fight_Time'] = fights_df['Fight_Time'].apply(remove_quotation)
    # Integer columns conversion
    fights_df['Sub. Att_1'] = fights_df['Sub. Att_1'].astype('Int8')
    fights_df['Sub. Att_2'] = fights_df['Sub. Att_2'].astype('Int8')
    fights_df['Rev._1'] = fights_df['Rev._1'].astype('Int8')
    fights_df['Rev._2'] = fights_df['Rev._2'].astype('Int8')
    # mm:ss to xxxx seconds
    fights_df['Ctrl_1'] = fights_df['Ctrl_1'].apply(
        parse_seconds_from_time).astype('Int32')
    fights_df['Ctrl_2'] = fights_df['Ctrl_2'].apply(
        parse_seconds_from_time).astype('Int32')

    fights_df.fillna({"Method Details": "No details provided"}, inplace=True)
    fights_df.fillna({"Referee": "Uknown referee"}, inplace=True)

    fights_df.dropna(subset=['KD_1'], inplace=True)
    # I already have a takedowns column + 37% of the data missing is too much to be imputed
    fights_df.drop(columns=['Td %_1', 'Td %_2'], inplace=True)


clean_fights_df()

fights_df.columns

print_missing_values(fights_df)

# I can not impute `['Wt.', 'Ht.','Reach']` by their means grouped by the current fighter `weight_class` because there is nearly no fights recorded for players with missing weight


missing_weight = fighters_df[fighters_df['Wt.'].isna()]['Full Name']
fights_of_missing_weight_1 = fights_df[fights_df['Fighter_1'].isin(
    missing_weight)]
fights_of_missing_weight_2 = fights_df[fights_df['Fighter_2'].isin(
    missing_weight)]
print(len(missing_weight))
print(len(fights_of_missing_weight_1)+len(fights_of_missing_weight_2))

# Doing this for all the columns I wanted to impute revealed a huge difference between the length of the two subsets.
# So maybe I'll consider dropping these fighters and their fights before training the model.


fights_df.to_csv('data/Fights.csv', index=False)
