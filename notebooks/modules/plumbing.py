'''
Created on Sep 2, 2019

@author: dan
'''
import csv


def infer_type(r):
    r_float = {}
    for (k,v) in r.items():
        try:
            z = float(v)
        except:
            z = v
        r_float[k] = z
    return r_float

def read_csv(file_name, delimiter=','):
    csvfile = open(file_name, newline='')
    reader = csv.DictReader(csvfile, delimiter=delimiter)
    return list(map(lambda r: infer_type(r), reader))

def write_csv(file_name, data, header=None):
    if header is None:
        header = list(data[0].keys())
    with open(file_name, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for n,r in enumerate(data):
            assert type(r) == dict,'expecting type(%s) == dict (%s)' % (r,type(r)) 
            r_str = dict(list(map(lambda x: (x[0],str(x[1])), r.items())))
            writer.writerow(r_str)
    return n

class FIFO(object):
    def __init__(self, depth):
        assert depth >= 1
        self.depth = depth
        self.values = []
    
    def update(self, x):
        self.values += [x]
        while len(self.values) > self.depth:
            self.values.pop(0)
        return self.values