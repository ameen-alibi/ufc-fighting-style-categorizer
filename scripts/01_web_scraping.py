#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-21T20:32:10.980Z
"""

# ### Scrape Fighters Data


from helpers import cached_request
from tqdm import tqdm
import warnings
from bs4 import MarkupResemblesLocatorWarning
import pandas as pd
import string
import bs4
import requests
import sys
import os

BASE_URL = "http://ufcstats.com/statistics/fighters"

# Remove BeautifulSoup warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# The page is organized by fighters names so I decided to
# Loop through different pages using an alphabets list
alphabets = list(string.ascii_lowercase)
for letter in alphabets:
    fighters_DOM_per_letter = cached_request(
        f"{BASE_URL}?char={letter}")
    soup = bs4.BeautifulSoup(fighters_DOM_per_letter, 'html.parser')
    pagination = soup.find('ul', class_='b-statistics__paginate')
    n_pasges = 1
    if pagination:
        n_pages = len(pagination.find_all('li')) if pagination else 1
    # Initialize the fighters dict
    if letter == 'a':
        headers = [th.get_text(strip=True)
                   for th in soup.select("table thead th")]
        fighters_data = {header.title(): [] for header in headers}

    # Not adding 1 to the number of list items because
    # there is a link for "all"
    # for i in range(1, n_pages+1):
    # The enumerated pages does not have all the fighters listed. I noticed that when I looked for Khabib & didn't find hime
    current_page = cached_request(
        f"{BASE_URL}?char={letter}&page=all")
    soup_1 = bs4.BeautifulSoup(current_page, 'html.parser')

    # Scraping fighters tabular data
    for row in soup_1.select("table tbody tr"):
        cells = row.find_all('td', class_='b-statistics__table-col')

        if len(cells) == 0:
            continue

        while len(cells) < len(headers):
            cells.append(None)

        for header, cell in zip(headers, cells):
            header = header.title()
            if cell:
                if header == 'Belt':
                    if len(cell.find_all()):
                        fighters_data[header].append(True)
                    else:
                        fighters_data[header].append(False)
                    continue
                fighters_data[header].append(cell.get_text(strip=True))
            else:
                fighters_data[header].append(None)

fighters_df = pd.DataFrame(fighters_data)

fighters_df.head()

fighters_df.to_csv('raw_data/raw_fighters.csv', index=False)

# ### Scrape Events


BASE_URL = "http://ufcstats.com/statistics/events/completed?page=all"

# This should not be a cashed request because the page is always updating but the url is the same
events_html = requests.get(BASE_URL).text


def get_event_data(event):
    first_td = event.select_one('td.b-statistics__table-col')
    event_link = first_td.select_one('a.b-link')
    event_name = event_link.get_text(strip=True)
    event_id = event_link['href'].split('/')[-1]
    event_date = first_td.select_one(
        'span.b-statistics__date').get_text(strip=True)

    location = event.select_one(
        'td.b-statistics__table-col_style_big-top-padding').get_text(strip=True)

    return [event_id, event_name, event_date, location]


events_soup = bs4.BeautifulSoup(events_html, 'html.parser')

# Get table headers
events_table_headers = [th.get_text(strip=True)
                        for th in events_soup.select('table thead th')]
# The first header is Name/Date so it is better to split it into separate name & date
events_table_headers = ['event_id']+events_table_headers[0].split(
    '/')+events_table_headers[1:]
events_dict = {header.title(): [] for header in events_table_headers}
events_rows = events_soup.select('table tbody tr.b-statistics__table-row')
# The first element is an empty row
events_rows.pop(0)
for event in events_rows:
    event_details = get_event_data(event)
    for header, cell in zip(events_table_headers, event_details):
        header = header.title()
        events_dict[header].append(cell)

# Storing events in a dataframe
events_df = pd.DataFrame(events_dict)
events_df.set_index('Event_Id', inplace=True)
events_df.head()

events_df.to_csv('raw_data/raw_events.csv')

# ### Scrape Events Details (Fights)


def get_fights_data(cells):
    result_flag = cells[0].get_text(strip=True)
    fighters = [a.get_text(strip=True)
                for a in cells[1].select("a.b-link")]
    kd = [p.get_text(strip=True) for p in cells[2].select("p")]
    strikes = [p.get_text(strip=True) for p in cells[3].select("p")]
    td = [p.get_text(strip=True) for p in cells[4].select("p")]
    sub = [p.get_text(strip=True) for p in cells[5].select("p")]
    weight_class = cells[6].get_text(strip=True)
    method = " ".join([p.get_text(strip=True)
                       for p in cells[7].select("p") if p.get_text(strip=True)])
    round_num = cells[8].get_text(strip=True)
    fight_time = cells[9].get_text(strip=True)

    return [result_flag, *fighters, *kd, *strikes, *
            td, *sub, weight_class, method, round_num, fight_time]


def extract_fights_from_event(event):
    first_td = event.select_one('td.b-statistics__table-col')
    event_link = first_td.select_one('a.b-link').get('href')
    # Visit each event link
    event_html = cached_request(event_link)
    event_soup = bs4.BeautifulSoup(event_html, "html.parser")
    # Fights in each event share the same fight_id
    event_id = event_link.split('/')[-1]
    # For each event, fights are arranged in tables
    fights = event_soup.select(
        "table.b-fight-details__table tbody tr.b-fight-details__table-row")
    return (event_id, fights)


# Initializing the dataframe dict
fights_headers = ["Fight_Id",
                  "Win/No Contest/Draw",
                  "Fighter_1",
                  "Fighter_2",
                  "KD_1",
                  "KD_2",
                  "STR_1",
                  "STR_2",
                  "TD_1",
                  "TD_2",
                  "SUB_1",
                  "SUB_2",
                  "Weight_Class",
                  "Method",
                  "Round",
                  "Fight_Time",
                  "Event_Id"
                  ]
fights_dict = {header: [] for header in fights_headers}

# I already have events_rows
# Loop through them
for event in events_rows:
    event_id, fight_rows = extract_fights_from_event(event)
    for fight_row in fight_rows:
        cells = fight_row.select('td')
        fight_info = get_fights_data(cells)
        fight_id = fight_row['data-link'].split('/')[-1]
        fight_info.insert(0, fight_id)
        fight_info.append(event_id)
        for key, val in zip(fights_headers, fight_info):
            fights_dict[key].append(val)

fights_df = pd.DataFrame(fights_dict)
fights_df.set_index('Fight_Id', inplace=True)
fights_df.head()

fights_df.to_csv('raw_data/raw_fights.csv')

# There are more details about fights at the "/fight-details" route


# ### Scrape Fights Details


def extract_cells(table, fight_dict={}, cols_to_ignore=0):
    cells = table.select('td')
    del cells[:cols_to_ignore]

    # Format table headers
    headers = [formatted_header.title() for header in table.select('th') for formatted_header in (
        f"{header.get_text(strip=True)}_1", f"{header.get_text(strip=True)}_2")]
    del headers[:cols_to_ignore*2]

    cells = [s for cell in cells for s in cell.stripped_strings]

    for header, cell in zip(headers, cells):
        fight_dict[header] = cell


def parse_fight_details(fight_html):
    fight_dict = {}

    soup = bs4.BeautifulSoup(fight_html, 'html.parser')

    # Get Result for each fighter W/L
    fighters_result_div = soup.select('.b-fight-details__person')
    fight_dict['Result_1'] = fighters_result_div[0].select_one(
        '.b-fight-details__person-status').get_text(strip=True)
    fight_dict['Result_2'] = fighters_result_div[1].select_one(
        '.b-fight-details__person-status').get_text(strip=True)

    fight_details_text = soup.select_one('.b-fight-details__content')
    paras = fight_details_text.find_all('p')

    if paras:
        # Get referee name and fight's time format
        first_para = paras[0]
        first_para_i_tags = first_para.find_all('i', recursive=False)
        fight_dict['Time Format'] = first_para_i_tags[3].get_text(
            strip=True).split(':')[1]
        fight_dict['Referee'] = first_para_i_tags[4].get_text(strip=True).split(':')[
            1]

        second_para = paras[1]
        # Remove the i tag in order to get only the text of the method details
        second_para.select_one('i').decompose()
        fight_dict['Method Details'] = second_para.get_text(strip=True)

    tables_soup = soup.select('table')
    # Ignoring this condition cost me waiting for 47m 28.6s then getting an error  :(
    if tables_soup:

        # Totals table
        totals_table = tables_soup[0]
        extract_cells(totals_table, fight_dict, cols_to_ignore=0)

        # Significant strikes table
        sig_str_table = tables_soup[2]
        extract_cells(sig_str_table, fight_dict, cols_to_ignore=3)

    return fight_dict


fights_list = []
events_total = len(events_rows)
# Adding tqdm to this loop made the waiting process less boring
for event in tqdm(
    events_rows,
    total=events_total,
    desc="Events",
    unit="event",
    position=0,
    leave=True,
    dynamic_ncols=True,
):
    event_id, fights = extract_fights_from_event(event)
    fights_total = len(fights)
    for fight in fights:
        try:
            fight_details_link = fight['data-link']
            fight_details_html = cached_request(fight_details_link)
            fight_dict = parse_fight_details(fight_details_html)
            fight_dict['Fight_Id'] = fight_details_link.split('/')[-1]
            fight_dict['Event_Id'] = event_id
            fights_list.append(fight_dict)
        except:
            continue

details_df = pd.DataFrame(fights_list)
details_df.set_index('Fight_Id', inplace=True)
details_df.to_csv('raw_data/raw_details.csv')

# ### Join Fights with their details


len(details_df) == len(fights_df)

\
# Join fights with their details on fight_id and keep it as the index
combined_df = fights_df.merge(details_df, on='Fight_Id', how='left')
combined_df.head()

# Using details_df fighter names because they are compatible with the stats order
# If fighter_1_x == fighter_1_y keep the stats as they are and just remove fighter_1_y and keep the fighter_1_x with renaming it to fighter_1
# If fighter_1_x == fighter_2_y . swap only the fights_df columns and keep the details_df columns:
#   col_1,col_2 =  col_2,col_1
cols_to_swap = ['KD_', 'STR_', 'TD_', 'SUB_']

for idx in combined_df.index:
    fighter_1_x = combined_df.loc[idx, 'Fighter_1_x']
    fighter_2_x = combined_df.loc[idx, 'Fighter_2_x']
    fighter_1_y = combined_df.loc[idx, 'Fighter_1_y']
    fighter_2_y = combined_df.loc[idx, 'Fighter_2_y']

    # If fighter_1_x matches fighter_2_y, we need to swap all _y columns
    if fighter_1_x == fighter_2_y and fighter_2_x == fighter_1_y:
        combined_df.loc[idx, ['Fighter_1_x', 'Fighter_2_x']] = combined_df.loc[idx, [
            'Fighter_2_x', 'Fighter_1_x']].values

        # Swap stats columns
        for col in cols_to_swap:
            col1 = f'{col}1'
            col2 = f'{col}2'
            combined_df.loc[idx, [col1, col2]
                            ] = combined_df.loc[idx, [col2, col1]].values

combined_df.drop(columns=['Fighter_1_y', 'Fighter_2_y'], inplace=True)
combined_df.rename(columns={
    'Fighter_1_x': 'Fighter_1',
    'Fighter_2_x': 'Fighter_2'
}, inplace=True)

combined_df.to_csv('raw_data/raw_fights_detailed.csv')
