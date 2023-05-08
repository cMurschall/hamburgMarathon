from datetime import timedelta
from bs4 import BeautifulSoup
import geopy.distance
import requests
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import csv
import re

from shapely import geometry, Point

sites = [
    {'sex': 'W', 'pages': 6},
    {'sex': 'M', 'pages': 15}
]


def marathon_scraping():
    with (
        open('runners.csv', mode='w') as runners_csvfile,
        open('times.csv', mode='w') as times_csvfile
    ):
        runners_writer = csv.writer(runners_csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, )
        times_writer = csv.writer(times_csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        runners_writer.writerow(['Start number', 'Sex', 'Finish time netto', 'Finish time brutto',
                                 'Fullname', 'Ageclass', 'Year of birth', 'Startgroup', 'Country'])

        times_writer.writerow(['Split', 'Time Of Day', 'Time', 'Diff', 'min/km', 'km/h', 'Place', 'Start number'])

        for site in sites:
            index = 1
            base_url = 'https://hamburg.r.mikatiming.de/2023'
            for page in range(1, site['pages'] + 1):
                sex = site["sex"]

                url = f'{base_url}?page={str(page)}&event=HML&event_main_group=custom.meeting.marathon&num_results=500&pid=list&search[sex]={sex}&search[age_class]=%25'
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                # rows = soup.find_all('li', {"class": ["list-group-item", "row"]})
                rows = soup.select('li.list-group-item.row:not(.list-group-header)')
                for row in rows:
                    try:
                        # place_primary = int(row.find('div', class_='place-primary').get_text())
                        # place_secondary = int(row.find('div', class_='place-secondary').get_text())
                        detail_link = row.find('a')['href']
                        full_name = row.find('a').get_text()

                        country = ''
                        result = re.search(r"(.+) \((.+)\)", full_name)
                        if result:
                            full_name = result.group(1)
                            country = result.group(2)

                        age_class = row.select('div.list-field.type-age_class')[0].get_text()
                        # times = row.select('div.list-field.type-time')
                        # finish_time = times[0].get_text().replace("Finish", "")
                        # brutto_time = times[1].get_text().replace("Brutto", "")
                        detail_url = f'{base_url}/{detail_link}'
                        req = requests.get(detail_url)
                        soup = BeautifulSoup(req.content, 'html.parser')

                        start_number = soup.select('td.f-start_no_text')[0].get_text()
                        finish_time_netto = soup.select('td.f-time_finish_netto')[0].get_text()
                        finish_time_brutto = soup.select('td.f-time_finish_brutto')[0].get_text()
                        year_of_birth = soup.select('td.f-birthday')[0].get_text()
                        start_group = soup.select('td.f-start_group')[0].get_text()

                        table = soup.select('table.table-condensed.table-striped')
                        df_list = pd.read_html(str(table[0]))
                        df = df_list[0]
                        df['Start number'] = start_number
                        times_writer.writerows(df.values)

                        runner_row = [start_number, sex, finish_time_netto, finish_time_brutto, full_name, age_class,
                                      year_of_birth, start_group, country]
                        runners_writer.writerow(runner_row)

                        print(f'{index}: added {sex} runner {full_name} from {country}')
                        index += 1
                    except Exception as e:
                        print(e)


def convert_to_seconds(timestamp):
    x = timestamp.split(":")
    return 60 * 60 * int(x[0]) + 60 * int(x[1]) + int(x[2])


def marathon_show_finish_distribution():
    df_runners = pd.read_csv('runners.csv')
    df_runners['Finish time netto'] = df_runners['Finish time netto'].apply(convert_to_seconds)
    df_runners['Finish time brutto'] = df_runners['Finish time brutto'].apply(convert_to_seconds)

    bins = np.arange(7200, 24388, 2 * 60)

    w = df_runners.loc[df_runners['Sex'] == 'W']
    m = df_runners.loc[df_runners['Sex'] == 'M']

    plt.title("Haspa 2023")
    plt.xlabel("Finish times")
    plt.ylabel("# runners")
    plt.hist(m['Finish time netto'], bins, label='men')
    plt.hist(w['Finish time netto'], bins, label='woman')

    plt.legend(loc='upper right')

    ticks = np.arange(2 * 60 * 60, 6.5 * 60 * 60, 0.5 * 60 * 60)
    plt.xticks(ticks, [timedelta(seconds=int(i)) for i in ticks], rotation=-20)

    plt.tight_layout()
    plt.show()


def marathon_analyze():
    # plt.figure(figsize=(5, 5), dpi=50)
    from fiona.drvsupport import supported_drivers
    supported_drivers['KML'] = 'rw'
    aussenalster = gpd.read_file("aussenalster.kml")
    binnenalster = gpd.read_file("binnenalster.kml")
    stadtpark = gpd.read_file("stadtpark.kml")
    marathon_route = gpd.read_file("marathon_route.kml")

    total = 0
    distances = []
    (x, y) = marathon_route.geometry[0].xy

    # distances.append((x[0], y[0]), 0)
    for previous, current in zip(zip(x, y), zip(x[1:], y[1:])):


        distance = geopy.distance.geodesic((previous[1], previous[0]), (current[1], current[0]))
        total += distance.km
        distances.append((current, distance.m))

    labels = [
        ((10.0185, 53.59607), "Stadtpark"),
        ((10.00708, 53.56442), "Alster"),
        ((9.96168, 53.55407), "St. Pauli"),
        ((9.91687, 53.55224), "Ottensen"),
        ((10.02107, 53.55244), "St. Georg"),
        ((9.97927, 53.56865), "Rotherbaum"),
        ((10.02811, 53.61267), "Alsterdorf"),
        ((9.99425, 53.5911), "Eppendorf"),
        ((10.03026, 53.57334), "Uhlenhorst"),
    ]

    # labels = gpd.GeoDataFrame(labels_df, crs="EPSG:4326", geometry=[Point(x) for x in labels_df['coordinates']])

    hamburg = stadtpark.plot(color="#89ff8a")
    aussenalster.plot(ax=hamburg, color="#89b8ff")
    binnenalster.plot(ax=hamburg, color="#89b8ff")
    marathon_route.plot(ax=hamburg, color="#cecece", linewidth=3)

    for label in labels:
        hamburg.annotate(label[1], xy=label[0], ha='center', fontsize=8, color="#2d2d2d")

    for percent in np.arange(0, 1, 0.1):
        (x, y) = marathon_route.geometry[0].interpolate(percent, normalized=True).xy
        circle = plt.Circle((x[0], y[0]), 0.01, zorder=10)
        hamburg.add_patch(circle)

    hamburg.text(9.9, 53.61, "12:01", bbox={'facecolor': 'white', 'alpha': 0.1, 'pad': 10})
    plt.show()


if __name__ == '__main__':
    marathon_analyze()
