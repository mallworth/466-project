import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
Test accuracy of predicting genre based on audio features using Random Forest
:param D: Pandas df
'''
def RF(D, drop_popularity=False):
    # Select numeric attributes (These correspond to audio features)
    D = D.dropna()
    gt = D.to_numpy()[:,-1]
    D = D.select_dtypes(include=['number'])

    if drop_popularity:
        D = D.drop(columns=["popularity"])

    idxs = np.random.permutation(len(D))
    D = D.to_numpy()[idxs]
    gt = gt[idxs]
    genres = sorted(list(set(gt)))

    avg_acc = 0
    confusion = {g+" "+p: 0 for g in genres for p in genres}

    # 10 fold cross val
    folds = np.array_split(idxs, 10)
    for i in range(10):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(10) if j != i])

        D_train, D_test, gt_train, gt_test = D[train], D[test], gt[train], gt[test]

        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(D_train, gt_train)
        predicted = rf.predict(D_test)

        # Update confusion matrix
        for g, p in zip(gt_test, predicted):
            confusion[g+" "+p] += 1
        acc = accuracy_score(gt_test, predicted)
        avg_acc += acc

    print(f"Average Accuracy: {avg_acc / 10}")
    
    # Format confusion matrix to be pretty printable
    con_matrix = pd.DataFrame(0, index=genres, columns=genres)

    for key, val in confusion.items():
        p, a = key.split(" ")
        # Row: Actual
        # Column: Predicted
        con_matrix.loc[a, p] = val

    print("Confusion Matrix")
    print(con_matrix)


if __name__ == '__main__':
    D = pd.read_csv("data/music_genre.csv")
    D = D.drop(columns=["instance_id"])

    # Uncomment to drop the popularity attribute before running Random Forest
    # RF(D, True)

    # Uncomment to run Random Forest
    # RF(D)

