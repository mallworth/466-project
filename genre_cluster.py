import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import rand_score

'''
Test effectivness of using audio features to cluster songs by genre using DBSCAN
:param D: Pandas df
'''
def dbscan(D):
    # Select numeric attributes (These correspond to audio features)
    gt = D.dropna(axis=0).to_numpy()[:,-1]
    D = D.select_dtypes(include=['number']).dropna(axis=0)
    max_rand = (0, None, None)

    for epsilon in np.arange(1, 8, 1):
        for samples in range(2, 10): 
            clusters = DBSCAN(eps=epsilon, min_samples=samples).fit(D)
            rand = rand_score(gt, clusters.labels_)

            if rand > max_rand[0]:
                max_rand = (rand, epsilon, samples)

    print(f'''Epsilon: {max_rand[1]}     Min Samples: {max_rand[2]}
Rand Score: {max_rand[0]}''')

'''
Test accuracy of predicting genre based on audio features using Random Forest
'''


if __name__ == '__main__':
    D = pd.read_csv("data/music_genre.csv")
    dbscan(D)
