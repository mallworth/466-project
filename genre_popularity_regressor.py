from random import choice
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd

'''
Test accuracy of predicting popularity on a dataset with heterogenous genre attribute
:param D: Pandas df
'''
def regress_hetero(D):
    D = D.sample(n=5000, random_state=42)
    gt = D["popularity"].dropna(axis=0).to_numpy()
    D = D.select_dtypes(include=['number']).dropna(axis=0)

    D_train, D_test, gt_train, gt_test = train_test_split(D, gt, test_size=0.2, random_state=42)

    regr = RandomForestRegressor(n_estimators=100)
    regr.fit(D_train, gt_train)

    predicted = regr.predict(D_test)
    print(f'''Random sample of 5000 tracks
MSE: {mean_squared_error(gt_test, predicted)}
MAE: {mean_absolute_error(gt_test, predicted)}''')

'''
Test accuracy of predicting popularity on a dataset with homogenous genre attribute
:param D: Pandas df
:param genre: Genre to select from D. Default None, randomly selects a genre
'''
def regress_homo(D, genre=None):
    if genre == None:
        genres = list(set(D["music_genre"]))
        g = choice(genres)
    else:
        g = genre

    D = D[D["music_genre"] == g]
    gt = D["popularity"].dropna(axis=0).to_numpy()
    D = D.select_dtypes(include=['number']).dropna(axis=0)

    D_train, D_test, gt_train, gt_test = train_test_split(D, gt, test_size=0.2, random_state=42)

    regr = RandomForestRegressor(n_estimators=100)
    regr.fit(D_train, gt_train)

    predicted = regr.predict(D_test)
    print(f'''Genre: {g}
MSE: {mean_squared_error(gt_test, predicted)}
MAE: {mean_absolute_error(gt_test, predicted)}''')
    return mean_squared_error(gt_test, predicted), mean_absolute_error(gt_test, predicted)

if __name__ == '__main__':
    D = pd.read_csv("data/music_genre.csv")
    D = D.drop(columns=["instance_id"])
    total_mse = 0
    total_mae = 0
    for genre in sorted(['Country','Hip-Hop', 'Jazz', 'Classical', 'Electronic', 'Rap', 'Rock', 'Anime', 'Alternative', 'Blues']):
        mse, mae = regress_homo(D, genre)
        total_mse += mse
        total_mae += mae

    print(f'''Avg MSE: {total_mse / 10}
Avg MAE: {total_mae / 10}''')

    # Uncomment to run the same model on a dataset of 5000 randomly selected tracks
    # regress_hetero(D)