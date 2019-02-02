'''
William Dreese
Movie Data Set Analysis
main.py

literally does everything needed to
replicate my results

runs with python 3.6
'''

import readnclean, ranking, pca, zipfs

if __name__=="__main__":

    rank_data = "rd2k30"
    pca_data  = "pcadataSn"

    print("Making Datasets")
    rank_data = readnclean.run(rank_data, pca_data)
    print("\nBeginning Ranking Model")
    top5 = ranking.run(rank_data)
    print("\n\nBegin PCA Analysis")
    pca.run(pca_data+".csv", top5, 10)
    print("\nZipf's Law Graph")
    print("(matplotlib stalls, so use ctrl-c)")
    zipfs.run("zipfsdata","ZipfsOutput",-1)
    
