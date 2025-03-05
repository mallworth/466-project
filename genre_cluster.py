import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

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
def RF(D):
    # Select numeric attributes (These correspond to audio features)
    gt = D.dropna(axis=0).to_numpy()[:,-1]
    D = D.select_dtypes(include=['number']).dropna(axis=0)
    D_train, D_test, gt_train, gt_test = train_test_split(D, gt, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=50, criterion="entropy")
    rf.fit(D_train, gt_train)
    predicted = rf.predict(D_test)
    # Getting ~53% accuracy currently
    # print(accuracy_score(gt_test, predicted))

'''
Test accuracy of predicting genre based on audio features using Gradient Boost
'''
def GB(D):
    # Select numeric attributes (These correspond to audio features)
    gt = D.dropna(axis=0).replace("Hip-Hop", "Hip-Hop/Rap").replace("Rap", "Hip-Hop/Rap").replace("Blues", "Blues/Jazz").replace("Jazz", "Blues/Jazz").to_numpy()[:,-1]
    D = D.select_dtypes(include=['number']).dropna(axis=0)
    D_train, D_test, gt_train, gt_test = train_test_split(D, gt, test_size=0.2, random_state=42)

    gb = GradientBoostingClassifier()
    gb.fit(D_train, gt_train)
    predicted = gb.predict(D_test)
    # ~57% accuracy wihtout merging genres
    # Accuracy of randomly guessing: 1/10 -> 10%
    # ~68% accuracy merging Hip-Hip + Rap, Blues + Jazz
    # Accuracy of randomly guessing: 1/8 -> 12.5%
    print(accuracy_score(gt_test, predicted))

'''
    Merging "similar" genres gives a 11% increase in accuracy,
    randomly guessing would give a 2.5% increase, can we conclude that
    merging yields higher accuracy because of this or is this not signifcant
    evidence to rule out accuracy increase due to fewer classes?
'''



if __name__ == '__main__':
    D = pd.read_csv("data/music_genre.csv")
    # dbscan(D)
    GB(D)
