'''
William Dreese
Movie Data Set Analysis
zipfs.py

uses matplotlib to produce a word-frequency
plot, to judge if the distrubution follows zipfs law
'''

import pandas as pd
import matplotlib
import matplotlib.pyplot as mpl
from operator import itemgetter

def sort_dict(d):
    return sorted(d.items(), key=itemgetter(1), reverse=True)

def run(read_file, write_image="temp",max_out=10000):
    
    data = pd.read_csv(read_file+".csv")
    data = data.to_dict()
    data = data["0"]
    data = sort_dict(data)
    lend = len(data)
    if max_out == -1: max_out=lend

    fig,ax = mpl.subplots()
    mpl.yscale("log")
    mpl.xscale("log")
    ax.plot([x for x in range(max_out)], [y[1] for y in data[:max_out]])
    ax.set(xlabel='Rank', ylabel='Freq', title=write_image)
    ax.grid()
    
    fig.savefig(write_image+".png")
    mpl.show()

if __name__=="__main__":
    run("zipfsdata","ZipfsOutput",-1)
