#!/usr/bin/env python
"""parses two ics files and calculates average start times for them depending on week"""
import pickle
import os.path
import math
from datetime import datetime
from typing import Callable, Optional
from collections import defaultdict
import statistics
import itertools as it

import numpy as np
from icalevents.icalevents import events as ical_events
from icalevents.icalparser import Event

npfit = np.polynomial.Polynomial.fit

def get_matching_events(calendar: str,
        include: Callable[[Event], bool],
        start: datetime,
        end:datetime) -> list[Event]:
    """parse and filter a list of events"""
    cal_events = ical_events(file=calendar, start = start, end = end)

    result: list[Event] = []
    for event in cal_events:
        if include(event):
            result.append(event)
    return result

calendar_names = ['asleep', 'melatonin']
routine_names = ['somna', 'vakna']

def read_events() -> list[Event]:
    """read ics files"""

    end = datetime.today()
    start = datetime(2022, 2, 13, 12, 00)

    events = get_matching_events('Calendar.ics',
            include= lambda x: x.summary in calendar_names,
            start=start,
            end=end)

    events += get_matching_events('routine.ics',
            include= lambda x: x.summary in routine_names,
            start=start,
            end=end)


    # avg & stdev time of piller, somna, asleep, vakna
    # avg & stdev difference between piller-somna, somna-asleep, asleep-vakna


    # merge both lists, sorted by start time
    events.sort(key=lambda e: e.start)
    return events



def main() -> None:
    """ get me some sleep data! """

    if os.path.isfile('events.pickle'):
        with open('events.pickle', 'rb') as file:
            all_events = pickle.load(file)
    else:
        all_events = read_events()
        with open('events.pickle', 'wb') as file:
            pickle.dump(all_events, file)

    data: dict[float, list[Event]] = defaultdict(list)
    all_names = calendar_names + routine_names


    curr_dosage = 0.0
    for event in all_events:
        if event.summary == 'melatonin':
            desc = event.description.split(' ')
            if desc.pop() != 'unblinded':
                curr_dosage = 0.0
                continue

            curr_dosage = float(desc[0])

        if event.summary in all_names:
            if curr_dosage:
                data[curr_dosage].append(event)
                if event.summary == 'vakna':
                    curr_dosage = 0.0


    data_by_type: dict[str, tuple[list[float],list[float]]] = defaultdict(
            lambda: ([], []))
    for dosage, events in data.items():
        event_by_summary, diffs = parse_events(events)
        print_stats(dosage, event_by_summary, diffs)
        for name,values in it.chain(event_by_summary.items(), diffs.items()):
            for value in values:
                data_by_type[name][0].append(dosage)
                data_by_type[name][1].append(value)

    print_linear_fit(data_by_type)

    save_as_csv(data)

def save_as_csv(data: dict[float, list[Event]]) -> None:
    """print raw calendar data to csv"""
    with open('data.csv', 'w', encoding='utf-8') as file:
        for dosage, events in data.items():
            for event in events:
                print(dosage, event.summary, round(datetime_to_seconds(event.start)),
                        sep=',', file=file)


def print_linear_fit(data: dict[str, tuple[list[float],list[float]]]) -> None:
    """do numpy fancy stuff"""
    print("Least squares fit between dosage & value")
    print(f"  {'name':15} {'slope':7} {'mean error'}")
    for name, (dosages, values) in data.items():
        _, offset = normalize(values)
        values = [(v - offset)%86400 for v in values]
        [_, slope], [residual, _, _, _] = npfit(dosages, values, deg=1, full=True)

        mean_error = (residual[0]/len(values))**0.5

        print(f"  {name:15} {'-' if slope < 0 else '+'}{format_time(abs(slope), True):6}"
        f" {format_time(mean_error, True)}")

def format_time(time: float, include_seconds: bool=False, format_as_time: bool=False) -> str:
    """format seconds in a day to a HH:MM[:SS] string"""
    res = ""
    hours: int = math.floor(time // 3600)
    if hours:
        res = f'{hours:02d}'
        if format_as_time:
            res += ':'
        else:
            res += 'h'
    if not include_seconds:
        minutes: int = round(time % 3600 // 60)
        res += f'{minutes:02d}'
        if not format_as_time:
            res += 'm'
        return res
    minutes = math.floor(time % 3600 // 60)
    seconds: int = round(time % 60)
    res += f'{minutes:02d}'
    if format_as_time:
        return res + f':{seconds:02d}'
    return res + f'm{seconds:02d}s'

def datetime_to_seconds(input_datetime: datetime) -> float:
    """convert a datetime into seconds into the day, taking into account timezone"""
    time = input_datetime.time()
    return time.hour * 3600 + time.minute*60 + time.second + time.microsecond

def parse_events(all_events: list[Event]) -> tuple[dict[str, list[float]],
        dict[str, list[float]]]:
    """print stats for a specified week"""
    event_by_summary : dict[str, list[float]] = defaultdict(list)
    diffs : dict[str, list[float]] = defaultdict(list)

    last_event: Optional[Event] = None
    for event in all_events:
        event_by_summary[event.summary].append(datetime_to_seconds(event.start) )
        if last_event:
            if (last_event.summary,event.summary) not in (
                ('vakna','melatonin'), ('somna', 'vakna')):
                diffs[f'{last_event.summary}-{event.summary}'].append(
                    (event.start - last_event.start).total_seconds()
                    )
        last_event = event

    return event_by_summary, diffs

def print_stats(dosage: float, event_by_summary: dict[str, list[float]],
        diffs: dict[str, list[float]]) -> None:
    """print stuff"""
    print(f"Dosage: {dosage} mg")
    print(f"  {'name':15} {'time':6}    {'stdev':6} {'n':>2}")
    for summary, times in event_by_summary.items():
        mean, stdev = mean_stdev(times)

        print(f"  {summary:15} {format_time(mean, False, True):6}"
                f" +- {format_time(stdev, stdev < 3600):6} {len(times):2}")

    for name, times in diffs.items():
        if len(times) < 2:
            print(name, len(times))
            continue
        mean, stdev = mean_stdev(times)
        print(f"  {name:15} {format_time(mean):6} +- {format_time(stdev, stdev < 3600):6}"
                f" {len(times):2}")

def normalize(times : list[float]) -> tuple[list[float], float]:
    """normalizes times in a day so they (hopefully) don't cross midnight.
    returns the offset needed to return the times to normal"""
    sorted_times = sorted(times)

    # find the biggest gap
    largest_diff = -1.0
    offset = -1.0
    for prev,curr in zip(sorted_times, sorted_times[1:]+sorted_times[:1]):
        diff = (curr-prev) % 86400
        if diff > largest_diff:
            largest_diff = diff
            offset = curr

    # shift all times by the offset
    return [(t - offset) % 86400 for t in sorted_times], offset


def mean_stdev(times : list[float]) -> tuple[float, float]:
    """calculate mean and stdev for a list of times in a day,
    will try to shift the times by 12 hours to compensate for them being
    across midnight"""
    norm_times, offset = normalize(times)

    mean = statistics.mean(norm_times)
    stdev = statistics.stdev(norm_times)

    return (mean+offset)%86400, stdev


if __name__ == '__main__':
    main()
