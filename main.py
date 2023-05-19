from datetime import timedelta
from bs4 import BeautifulSoup
import geopy.distance
import requests
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import interpolate
import csv
import re
from tqdm import trange
import random
import os







from matplotlib.patches import Ellipse
from shapely import geometry, Point

sites = [
    {'sex': 'W', 'pages': 6},
    {'sex': 'M', 'pages': 15}
]


def scrape_marathon_data():
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


def analyze_marathon():
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

    path_offset_distance = 0.0005
    runner_paths = [marathon_route.geometry[0].offset_curve(i) for i in
                    np.linspace(-path_offset_distance, path_offset_distance, 4)]
    runner_paths.append(marathon_route.geometry[0])
    runner_paths_dict = {i: random.choice(runner_paths) for i in df_positions.index}

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)

    # labels = gpd.GeoDataFrame(labels_df, crs="EPSG:4326", geometry=[Point(x) for x in labels_df['coordinates']])
    i = 1
    while os.path.exists(f"haspa{i}.mp4"):
        i += 1

    with writer.saving(fig, f"haspa{i}.mp4", dpi=300):
        count_frames = df_positions.shape[1]

        # count_frames = 200
        for frame in trange(count_frames):

            ax.set_xlim([9.882240795, 10.049447505])
            ax.set_ylim([53.539870385, 53.629993715])


            column = df_positions.iloc[:, frame]

            stadtpark.plot(ax=ax, color="#89ff8a")

            # keep tilt ratio for drawing circles later
            # dx, dy = ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0))

            aussenalster.plot(ax=ax, color="#89b8ff")
            binnenalster.plot(ax=ax, color="#89b8ff")
            marathon_route.plot(ax=ax, color="#cecece", linewidth=7)

            for label in labels:
                ax.annotate(label[1], xy=label[0], ha='center', fontsize=8, color="#2d2d2d")

            # draw runners
            runner_percentages = column[column.between(0, 42000)] / 42000

            def find_position_of_runner(i):
                path = runner_paths_dict[i[0]]
                return path.interpolate(i[1], normalized=True).xy

            pos = pd.Series(runner_percentages.reset_index().apply(find_position_of_runner, axis=1).values,
                            index=runner_percentages.index)
            # pos = runner_percentages.apply(lambda i: marathon_route.geometry[0].interpolate(i, normalized=True).xy)

            x_pos, y_pos = zip(*pos)

            # pos_male = pos[[not 'F' in s for s in pos.index]]
            # pos_female = pos[['F' in s for s in pos.index]]

            # x_pos_male, y_pos_male = zip(*pos_male)
            # x_pos_female, y_pos_female = zip(*pos_female)

            color_male = "#00C4AA"
            color_female = "#8700F9"
            #ax.scatter(x_pos_male, y_pos_male, c=color_male, s=2, zorder=10)
            #ax.scatter(x_pos_female, y_pos_female, c=color_female, s=2, zorder=11)

            colors = [color_female if "F" in i else color_male for i in runner_percentages.index]
            ax.scatter(x_pos, y_pos, c='k', alpha=0.5, s=2, zorder=10)

            font = {'family': 'sans-serif', 'size': 20}
            # box = {'facecolor': 'white', 'alpha': 0.1, 'pad': 10}

            ax.text(9.9, 53.61, column.name.strftime("%H:%M"), fontdict=font)

            #plt.show()
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


def interpolate_runner_positions():
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

    start_time = pd.to_datetime('2023-04-23 09:30', format='%Y-%m-%d %H:%M')
    time_stamps = pd.date_range(start=start_time, end=max(df_times["Time"]), freq="10S")
    start_timestamp = np.int64(start_time.asm8.astype(np.int64) / 10 ** 9)


    positions_dict = {}
    error_count = 0

    for index, row in df_runners.iterrows():
        times = df_times[df_times["Start number"] == str(row["Start number"])].sort_values(by=['Timestamp'])
        try:
            valid_values = times[times["Timestamp"] > 0][["Timestamp", "Distance"]]

            x = np.concatenate(([start_timestamp], valid_values["Timestamp"]))
            y = np.concatenate(([0], valid_values["Distance"]))
            # cs = interpolate.CubicSpline(x, y)
            cs = interpolate.interp1d(x, y, fill_value='extrapolate')
            positions = cs(time_stamps.values.astype('int64') // 10 ** 9)
            positions_dict[str(row["Start number"])] = positions
        except Exception as e:
            error_count += 1
            print(e)
            print("Error: " + row["Start number"])
            print(f"{len(times['Timestamp']) - len(times['Distance'])}")
            print("---------")

            # positions_dict[str(row["Start number"])] = np.zeros(len(time_stamps))
            # this should always return the 0 distance

    print("Error count " + str(error_count))
    df_positions = pd.DataFrame.from_dict(positions_dict, orient='index', columns=time_stamps)
    df_positions.to_pickle("positions.pkl")


if __name__ == '__main__':
    # scrape_marathon_data();
    interpolate_runner_positions()
    analyze_marathon()
    print("done")
