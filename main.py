from datetime import timedelta
from bs4 import BeautifulSoup
import geopy.distance
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.interpolate import CubicSpline
import csv
import re
from tqdm import tqdm , trange

from matplotlib.patches import Ellipse
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
        total += distance.m
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


    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Haspa Marathon 2023', artist='Christian Murschall', comment='Zu viel Zeit')
    writer = FFMpegWriter(fps=24, metadata=metadata, bitrate=900)

    df_positions = pd.read_pickle("positions.pkl")


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # labels = gpd.GeoDataFrame(labels_df, crs="EPSG:4326", geometry=[Point(x) for x in labels_df['coordinates']])

    with writer.saving(fig, "haspa.mp4", dpi=300):
        count_frames = df_positions.shape[1]
        count_frames = 1000
        for i in trange(count_frames):
            column = df_positions.iloc[:, i]

            stadtpark.plot(ax=ax, color="#89ff8a")

            # keep tilt ratio for drawing circles later
            dx, dy = ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0))

            aussenalster.plot(ax=ax, color="#89b8ff")
            binnenalster.plot(ax=ax, color="#89b8ff")
            marathon_route.plot(ax=ax, color="#cecece", linewidth=7)

            for label in labels:
                ax.annotate(label[1], xy=label[0], ha='center', fontsize=8, color="#2d2d2d")

            # draw runners
            runner_percentages = column[column.between(0, 42000)] / 42000
            for percent in runner_percentages:
                (x, y) = marathon_route.geometry[0].interpolate(percent, normalized=True).xy

                # calculate asymmetry of x and y direction
                point_size = .0008
                maxd = min(dx, dy)
                width = point_size* maxd / dy
                height = point_size * maxd / dx

                circle = Ellipse((x[0], y[0]), width, height, zorder=10)
                ax.add_patch(circle)

            ax.text(9.9, 53.61, column.name.strftime("%H:%M"), bbox={'facecolor': 'white', 'alpha': 0.1, 'pad': 10})

            writer.grab_frame()
            ax.clear()



def split_to_distance(row):
    total_length = 42000

    if "10km" in row:
        return 10_000
    if "15km" in row:
        return 15_000
    if "20km" in row:
        return 20_000
    if "25km" in row:
        return 25_000
    if "30km" in row:
        return 30_000
    if "35km" in row:
        return 35_000
    if "40km" in row:
        return 40_000
    if "Halb" in row:
        return total_length / 2
    if "Finish" in row:
        return total_length
    # ha ha = since 5km is in all uneven cases we need to put it at the bottom
    if "5km" in row:
        return 5_000
    return 0


def get_interpolate_runner_positions():
    df_runners = pd.read_csv('runners.csv')
    df_runners['Finish time netto'] = df_runners['Finish time netto'].apply(convert_to_seconds)
    df_runners['Finish time brutto'] = df_runners['Finish time brutto'].apply(convert_to_seconds)

    df_times = pd.read_csv("times.csv")

    df_times["Distance"] = df_times["Split"].apply(split_to_distance)
    df_times["Time"] = pd.to_datetime(df_times["Time Of Day"], format='%H:%M:%S', errors="coerce").apply(
        lambda dt: dt.replace(day=23, month=4, year=2023))
    df_times["Timestamp"] = df_times["Time"].astype('int64') // 10 ** 9

    # female runners have a F in the start number, males don´t. We convert the columns to string so we can compare
    df_times["Start number"] = df_times["Start number"].astype(str)
    df_runners["Start number"] = df_runners["Start number"].astype(str)

    time_stamps = pd.date_range(start=min(df_times["Time"]), end=max(df_times["Time"]), freq="10S")

    positions_dict = {}
    error_count = 0

    for index, row in df_runners.iterrows():
        times = df_times[df_times["Start number"] == str(row["Start number"])].sort_values(by=['Timestamp'])
        try:
            valid_values = times[times["Timestamp"] > 0][["Timestamp", "Distance"]]
            cs = CubicSpline(valid_values["Timestamp"], valid_values["Distance"])
            positions = cs(time_stamps.values.astype('int64') // 10 ** 9)
            positions_dict[str(row["Start number"])] = positions
        except Exception as e:
            error_count += 1
            print(e)
            print("Error: " + row["Start number"])
            print(f"{len(times['Timestamp']) - len(times['Distance'])}")
            print("---------")

            positions_dict[str(row["Start number"])] = np.zeros(len(time_stamps))
            # this should always return the 0 distance

    print("Error count " + str(error_count))
    df_positions = pd.DataFrame.from_dict(positions_dict, orient='index', columns=time_stamps )
    df_positions.to_pickle("positions.pkl")


def matplotlib_movie_test():


    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=24, metadata=metadata, bitrate=900)

    fig = plt.figure()
    l, = plt.plot([], [], 'k-o')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    x0, y0 = 0, 0

    with writer.saving(fig, "writer_test.mp4", dpi=100):
        for i in range(100):
            x0 += 0.1 * np.random.randn()
            y0 += 0.1 * np.random.randn()
            l.set_data(x0, y0)
            writer.grab_frame()


if __name__ == '__main__':
    # get_interpolate_runner_positions()
    marathon_analyze()
    print("done")
