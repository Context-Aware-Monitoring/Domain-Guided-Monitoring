from collections import namedtuple

EvDef = namedtuple('EvDef', ['type', 'value'])

class EventDefinitionMap:
    def __init__(self, top_dt, end_dt, timestamp_column):
        self.top_dt = top_dt
        self.end_dt = end_dt
        self.timestamp_column = timestamp_column
        self._emap = {}
        self._ermap = {}

    def __len__(self):
        return len(self._emap)

    def _eids(self):
        return self._emap.keys()

    def _next_eid(self):
        eid = len(self._emap)
        while eid in self._emap:
            eid += 1
        else:
            return eid

    def get_evdef(self, eid) -> EvDef:
        return self._emap[eid]

    def get_eid(self, evdef):
        return self._ermap[evdef]

    def process_row(self, columns, row):
        row_eids = []
        for column in columns:
                if column == self.timestamp_column:
                    continue
                value = row[column]
                if value == "":
                    continue
                d = {
                    "type": column,
                    "value": row[column],
                }

                evdef = EvDef(**d)

                if evdef in self._ermap:
                    row_eids.append(self._ermap[evdef])
                else:
                    eid = self._next_eid()
                    self._emap[eid] = evdef
                    self._ermap[evdef] = eid
                    row_eids.append(eid)
        return row_eids