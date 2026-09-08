"""Explicit NCU raw wide-CSV compatibility; no GPU work or byte rewriting."""
import csv
import io
import math

IDENTITY = ('ID', 'Process ID', 'Kernel Name')

def is_wide(raw):
    return any(all(k in row for k in IDENTITY) and 'Metric Name' not in row
               for row in csv.reader(io.StringIO(raw)))

def decode_wide(raw, metrics, kernel_identifier):
    if '==ERROR==' in raw or 'ERR_NVGPUCTRPERM' in raw:
        raise ValueError('profiler error in exported data')
    rows = list(csv.reader(io.StringIO(raw)))
    headers = [i for i, row in enumerate(rows) if all(k in row for k in IDENTITY)]
    if len(headers) != 1:
        raise ValueError('exactly one raw header required')
    i = headers[0]; header = rows[i]
    if len(header) != len(set(header)) or any(not k for k in header):
        raise ValueError('duplicate or empty raw column')
    if len(metrics) != len(set(metrics)) or not set(metrics).issubset(header):
        raise ValueError('missing or duplicate requested metrics')
    tail = [row for row in rows[i+1:] if row]
    if len(tail) != 2 or any(len(row) != len(header) for row in tail):
        raise ValueError('one units row and exactly one complete launch required')
    units, values = (dict(zip(header, row)) for row in tail)
    if any(units[k] for k in IDENTITY):
        raise ValueError('invalid units row')
    if not values['ID'].isdigit() or not values['Process ID'].isdigit() or int(values['Process ID']) <= 0:
        raise ValueError('invalid launch identity')
    if not kernel_identifier or kernel_identifier not in values['Kernel Name']:
        raise ValueError('wrong native kernel identity')
    parsed = {}
    for metric in metrics:
        try: value = float(values[metric].strip().replace(',', ''))
        except ValueError: raise ValueError('nonnumeric hardware metric: ' + metric) from None
        if not math.isfinite(value) or value < 0:
            raise ValueError('nonfinite or negative hardware metric')
        unit = units[metric]
        if metric in ('dram__bytes_read.sum', 'dram__bytes_write.sum') and unit not in ('byte', 'bytes'):
            raise ValueError('DRAM metric must use base bytes')
        if metric == 'sm__cycles_elapsed.avg' and (unit not in ('cycle', 'cycles') or value <= 0):
            raise ValueError('positive SM cycles required')
        if metric == 'gpu__time_duration.sum':
            scale = {'s':1., 'ms':1e-3, 'us':1e-6, 'ns':1e-9, 'second':1., 'seconds':1., 'msecond':1e-3, 'usecond':1e-6, 'nsecond':1e-9}.get(unit)
            if scale is None or not math.isfinite(value*scale) or value*scale <= 0:
                raise ValueError('positive supported duration required')
        parsed[metric] = {'value':value, 'unit':unit}
    result = {'kernel_name':values['Kernel Name'], 'process_id':values['Process ID'],
              'kernel_id':values['ID'], 'metrics':parsed}
    metadata = {'format':'ncu_raw_wide_v1', 'column_count':len(header),
                'extra_columns':[k for k in header if k not in metrics and k not in IDENTITY],
                'data_rows':1}
    return result, metadata
