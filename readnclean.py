'''
William Dreese
Movie Dataset Anaylsis
readnclean.py

performs initial reading and preprocessing of data
for all my expirements and tests
'''

import sys
import numpy as np
import pandas as pd
from operator import itemgetter

def sort_dict(d):
    return sorted(d.items(), key=itemgetter(1), reverse=True)

''' read in data, drop completly unneeded columns '''
def get_data():
    data = pd.read_csv("movie_data.csv")
    data = data.drop("id",axis=1)
    data = data.drop("title",axis=1)
    data = data.drop("runtime",axis=1)
    return data

''' finds and returns the year for each date object '''
def clean_date(data):
    def mapper(x):
        xr = x[0]
        if '-' in xr: xr = xr[:xr.find('-')]
        return int(xr)
    data['release_date'] = data.apply(mapper,axis=1)
    return data

''' opens dataset ''' 
def get_genres(file_name):
    h = open(file_name,"r")
    genres = (h.readline().split(","))[:-1]
    h.close()
    return genres

''' use sets to find unique words, then count them with a dict '''
def make_pca(out_name,genre_csv="rd2k30tn",year=2000,ref_min=200,ref_max=3000,small=False):

    ''' dataset adjustments ''' 
    data = get_data()
    if small:
        data = data.dropna()
        data = data.drop("box_office_revenue",axis=1)
    else:
        data = data.drop("box_office_revenue",axis=1)
        data = data.dropna()
    data['release_date'] = data['release_date'].astype('str')
    data['summary']      = data['summary'].astype('str')

    ''' clean and drop dates before 2000 (not used in ranking model) '''
    data = clean_date(data)
    data = data.drop(data[data.release_date < year].index)
    data = data.drop("release_date",axis=1)

    unique = dict()
    genres = get_genres(genre_csv+".csv")
    unique_genre = dict()
    for gen in genres: unique_genre[gen] = dict()

    ''' these two functions help clean the movie summaries '''
    def real_letter(c):
        if c > 64 and c < 91: return True
        elif c > 96 and c < 123: return True
        elif c == 32: return True
        return False

    def clean_words(x):
        summ = x[1]
        summ = ''.join(c for c in summ if real_letter(ord(c)))
        summ = summ.upper()
        summ = summ.split()
        for word in summ:
            if word in unique: unique[word] += 1
            else: unique[word] = 1
        return ' '.join(s for s in summ)

    data["summary"] = data.apply(clean_words,axis=1)

    word_freq = pd.DataFrame.from_dict(unique,orient="index")
    word_freq.to_csv("zipfsdata.csv")

    ''' some word frequency analysis '''
    keycount = 0
    wordcount = 0
    wordsleft = 0
    under = list()
    buckets = dict({10: 0, 30: 0, 100: 0, 200: 0, 500: 0, 1000: 0, 5000: 0, 1000000:0})
    for key in unique.keys():
        keycount += 1
        wordcount += unique[key]
        if unique[key] < ref_min: under.append(key)
        elif unique[key] > ref_max: under.append(key)
        else: wordsleft += 1
        for b in buckets.keys():
            if unique[key] <= b:
                buckets[b] += 1
                break
    for un in under: del unique[un]

    print("Unique Words: "+str(keycount))
    print("Entire Count: "+str(wordcount))
    print("Words Left: "+str(wordsleft))
    print(buckets)
    unique_sorted = sort_dict(unique)

    for gen in genres:
        for wd in unique: unique_genre[gen][wd] = 0.0

    gen_cnt = dict()
    for gen in genres: gen_cnt[gen] = 0
        
    def get_genre_count(x):
        xg = x[0].split(',')
        for gen in xg:
            gen = gen[gen.find('"')+1:]
            gen = gen[:gen.find('"')]
            if gen in genres: gen_cnt[gen] += 1
    
    def create_dataset(x):
        summ = x[1].split()
        xg = x[0].split(',')
        gens = list()
        gc = 0.0
        for gen in xg:
            gen = gen[gen.find('"')+1:]
            gen = gen[:gen.find('"')]
            gens.append(gen)
            gc += 1.0
        for wd in summ:
            for gen in gens:
                if wd in unique and gen in genres:
                    unique_genre[gen][wd] += (1.0 / gc) / gen_cnt[gen]
                    
    data.apply(get_genre_count,axis=1)
    data.apply(create_dataset,axis=1)
    print(sort_dict(gen_cnt)[:10])
    pca_data = pd.DataFrame(0, index=genres, columns=unique.keys())

    for gen in genres:
        for wd in unique.keys():
            pca_data.loc[gen,wd] = unique_genre[gen][wd]

    pca_data.to_csv(out_name)

''' makes a lightly pre-pro version of the original dataset
    for other functions to use '''
def get_pre_data():
    data = get_data()
    data = data.drop("summary",axis=1)
    data = data.dropna() #drop data with NaNs, since remaining data is needed
    data['release_date'] = data['release_date'].astype('str')
    data['genres']       = data['genres'].astype('str')

    data = clean_date(data)

    genres = dict()

    def get_genres(x):
        xg = x[2].split(',')
        for gen in xg:
            gen = gen[gen.find('"')+1:]
            gen = gen[:gen.find('"')]
            if gen not in genres: genres[gen] = 1
            else: genres[gen] += 1

    data.apply(get_genres,axis=1)
    data = data.reset_index(drop=True)

    return data,genres

''' normalize a column using normal or min/max '''
def norm_col(csv="ranking_data",col="box",minmax=False):

    data = pd.read_csv(csv+".csv")
    if minmax: data[col] = (data[col]-data[col].min()) / (data[col].max()-data[col].min())
    else: data[col] = (data[col] - data[col].mean()) / data[col].std()
    data.to_csv(csv+"n.csv",index=False)

''' if mention total < thres, drop that col
after, make sure all movies have atleast one genre '''
def thres_genres(csv="ranking_data",thres=10):

    data = pd.read_csv(csv+".csv")
    for col in list(data):
        if (data[col].sum() < thres) and col != "box":
            data = data.drop(col,axis=1)
    data['sums'] = data.sum(axis=1)-data['box']
    data = data.drop(data[data.sums < 1].index)
    data = data.reset_index(drop=True)
    data = data.drop('sums',axis=1)
    data.to_csv(csv+"t.csv",index=False)

''' creates dataset ''' 
def make_ranking_data(csv="ranking_data",year=2000):

    data,genres = get_pre_data()

    ''' make new (ids X genres) dataframe '''
    keylist = list(genres.keys())
    keydict = dict()
    for k in range(len(keylist)): keydict[keylist[k]] = k
    gendata = pd.DataFrame(0, index=np.arange(len(data)), columns=list(keylist))

    gendata['box'] = data['box_office_revenue']
    gendata = gendata.drop(gendata[data.release_date < year].index)
    data    = data.drop(data[data.release_date < year].index)
    data    = data.reset_index(drop=True)
    gendata = gendata.reset_index(drop=True)
    
    def make_one_hot(x):
        xg = x[2].split(',')
        for gen in xg:
            gen = gen[gen.find('"')+1:]
            gen = gen[:gen.find('"')]
            gendata.iloc[x[3],keydict[gen]] = 1
    
    data['ind'] = data.index
    data.apply(make_one_hot,axis=1)

    gendata.to_csv(csv+".csv", index=False)

''' normalizes created ranking dataset ''' 
def make_rank(saveas="rd2k", yr=2000, th=30, mm=False):
    make_ranking_data(csv=saveas, year=yr)
    thres_genres(csv=saveas, thres=th)
    norm_col(csv=saveas+"t", minmax=mm)
    return saveas+"tn"

def run(rankcsv,pcacsv):
    s=make_rank(saveas=rankcsv)
    make_pca(out_name=pcacsv+".csv",genre_csv=s,small=True)
    return s
