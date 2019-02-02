'''
William Dreese
Movie Data Set Analysis
pca.py

the PCA model used in identifying key
words in the top 5 genres
'''

import numpy as np
import pandas as pd
import matplotlib
from sklearn.decomposition import PCA
from operator import itemgetter

def sort_dict(d):
    return sorted(d.items(), key=itemgetter(1), reverse=True)

def make_dict(data, gens, vals):
    data.set_index(gens).to_dict()[vals]

def run(data_file, top5, comps):

    data = pd.read_csv(data_file)
    genres = data.columns[0]
    
    data_u = data.drop(data.columns[0],axis=1)

    ''' create PCA model, train '''
    model = PCA(n_components=comps)
    true = model.fit_transform(data_u)
    inv = model.inverse_transform(np.eye(comps))

    ''' get variance rankings of each word used '''
    cols = list(data_u)
    contribs = dict()
    feature_contribs = inv.mean(axis=0)
    for k in range(len(cols)):
        contribs[cols[k]] = feature_contribs[k]
    rank_var = sort_dict(contribs)

    ''' this alg finds the most chracteristic words of a
        a genre by cross-ref top words of each genre '''
    top_words = dict()
    for gen in top5: top_words[gen] = list()
    for k in range(100):
        col = rank_var[k][0]
        d = data.set_index(genres).to_dict()[col]
        s_d = sort_dict(d)[:20]
        for gen in top5:
            if gen in [s[0] for s in s_d]: top_words[gen].append(col)

    ''' print out results '''
    for wd in top5:
        print(wd.upper())
        print(', '.join(s.lower() for s in (top_words[wd])[:10]))
    
