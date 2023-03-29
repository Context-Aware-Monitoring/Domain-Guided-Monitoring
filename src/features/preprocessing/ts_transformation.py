# Based on the code from https://github.com/amulog/logdag

from src.features.preprocessing.evdef import EventDefinitionMap
import datetime
import pandas as pd
import numpy as np
import dataclass_cli
import dataclasses
from typing import List, Dict, Tuple, Set, Union
import re

@dataclass_cli.add
@dataclasses.dataclass
class TimeSeriesTransformerConfig:
    bin_size: str = '00:00:60'
    bin_overlap: str = '00:00:50'
    timestamp_column: str = '@timestamp'

def parse_timedelta(stamp: str) -> Union[datetime.timedelta,str]:
    if 'day' in stamp:
        m = re.match(r'(?P<d>[-\d]+) day[s]*, (?P<h>\d+):'
                     r'(?P<m>\d+):(?P<s>\d[\.\d+]*)', stamp)
    else:
        m = re.match(r'(?P<h>\d+):(?P<m>\d+):'
                     r'(?P<s>\d[\.\d+]*)', stamp)
    if not m:
        return ''

    time_dict = {key: float(val) for key, val in m.groupdict().items()}
    if 'd' in time_dict:
        return datetime.timedelta(days=time_dict['d'], hours=time_dict['h'],
                         minutes=time_dict['m'], seconds=time_dict['s'])
    else:
        return datetime.timedelta(hours=time_dict['h'],
                         minutes=time_dict['m'], seconds=time_dict['s'])

class TimeSeriesTransformer():

    def __init__(self, config: TimeSeriesTransformerConfig):
        self.config = config
        self.bin_size = parse_timedelta(self.config.bin_size)
        self.bin_overlap = parse_timedelta(self.config.bin_overlap)

    def transform_time_series_to_events(self, data_df: pd.DataFrame, relevant_columns: Set[str]):
        data_df.drop(labels=[x for x in data_df.columns if x not in relevant_columns], axis=1, inplace=True)
        data_df = data_df.sort_values(by=self.config.timestamp_column).reset_index(drop=True).fillna("")
        top_dt, end_dt = self._get_date_range(data_df)
        evmap, evdict = self._create_maps(data_df, top_dt, end_dt)
        data = self._event2stat(evdict, top_dt, end_dt)
        dm = np.array([d for _, d in sorted(data.items())]).transpose()
        df = pd.DataFrame(dm)
        return df, evmap

    def _label(self, dt_range: Tuple[datetime.datetime, datetime.datetime], duration) -> List[datetime.datetime]:
        top_dt, end_dt = dt_range
        l_label = []
        temp_dt = top_dt
        while temp_dt < end_dt:
            l_label.append(temp_dt)
            temp_dt += duration
        l_label.append(end_dt)
        return l_label

    def _discretize(self, l_dt, l_label, method):

        def return_empty(bin_num):
            if method in ("count", "binary"):
                return [0] * bin_num
            elif method == "datetime":
                return [[] for i in range(bin_num)]
            else:
                raise NotImplementedError(
                    "Invalid method name ({0})".format(method))

        def init_tempobj():
            if method == "count":
                return 0
            elif method == "binary":
                return 0
            elif method == "datetime":
                return []
            else:
                raise NotImplementedError(
                    "Invalid method name ({0})".format(method))

        def update_tempobj(temp):
            if method == "count":
                return temp + 1
            elif method == "binary":
                return 1
            elif method == "datetime":
                temp.append(new_dt)
                return temp
            else:
                raise NotImplementedError(
                    "Invalid method name ({0})".format(method))

        bin_num = len(l_label) - 1
        l_dt_temp = sorted(l_dt)
        if len(l_dt_temp) <= 0:
            return_empty(bin_num)

        iterobj = iter(l_dt_temp)
        try:
            new_dt = next(iterobj)
        except StopIteration:
            raise ValueError("Not empty list, but failed to get initial value")
        while new_dt < l_label[0]:
            try:
                new_dt = next(iterobj)
            except StopIteration:
                return_empty(bin_num)

        ret = []
        stop = False
        for label_dt in l_label[1:]:
            temp = init_tempobj()
            if stop:
                ret.append(temp)
                continue
            while new_dt < label_dt:
                temp = update_tempobj(temp)
                try:
                    new_dt = next(iterobj)
                except StopIteration:
                    # "stop" make data after label term be ignored
                    stop = True
                    break
            ret.append(temp)
        return ret

    def _autodiscretize_with_slide(self, l_dt, dt_range):
        slide = self.bin_size - self.bin_overlap
        top_dt, end_dt = dt_range
        slide_width = max(int(self.bin_size.total_seconds() / slide.total_seconds()), 1)
        l_top = self._label((top_dt, end_dt), slide)[:-1]
        l_end = [min(t + self.bin_size, end_dt) for t in l_top]

        ret = []
        noslide = self._discretize(l_dt, l_top + [end_dt], method='datetime')
        for i, bin_end in enumerate(l_end):
            l_dt_temp = []
            for b in noslide[i:i+slide_width]:
                l_dt_temp.extend([dt for dt in b if dt <= bin_end])
        
            if len(l_dt_temp) > 0:
                ret.append(1)
            else:
                ret.append(0)

        return ret

    def _event2stat(self, evdict: Dict, top_dt: datetime.datetime, end_dt: datetime.datetime):
        d_stat = {}
        labels = self._label((top_dt, end_dt), self.bin_size)
        for eid, l_ev in evdict.items():
            if len(l_ev) == 0:
                continue
            if self.bin_overlap == datetime.timedelta(seconds = 0):
                val = self._discretize(l_ev, labels, method="binary")
            else:
                val = self._autodiscretize_with_slide(l_ev, dt_range = (top_dt, end_dt))
            if val is not None:
                d_stat[eid] = val
        return d_stat

    def _create_maps(self, data_df: pd.DataFrame, top_dt: datetime.datetime, end_dt: datetime.datetime):
        evmap = EventDefinitionMap(top_dt=top_dt, end_dt=end_dt, timestamp_column=self.config.timestamp_column)
        evdict = {}
        def row_func(row):
            row_eids = evmap.process_row(data_df.columns, row)
            timestamp = row[self.config.timestamp_column]
            for eid in row_eids:
                if eid in evdict:
                    evdict[eid].append(timestamp)
                else:
                    evdict[eid] = [timestamp]
            return row
        data_df.apply(row_func, axis=1)        
        return evmap, evdict

    def _get_date_range(self, data_df: pd.DataFrame):
        min_dt: datetime.datetime = data_df[self.config.timestamp_column].iloc[0].to_pydatetime()
        max_dt: datetime.datetime = data_df[self.config.timestamp_column].iloc[-1].to_pydatetime()
        top_dt = datetime.datetime.combine(min_dt.date(), datetime.time(hour=min_dt.hour)).replace(tzinfo=min_dt.tzinfo)
        end_dt = datetime.datetime.combine(max_dt.date(), datetime.time(hour=max_dt.hour, minute=max_dt.minute + 1)).replace(tzinfo=min_dt.tzinfo)
        return top_dt, end_dt
