import numpy as np
import pandas as pd
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# xgboostによるモデル
class Model1Xgb:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        # params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
        #           'eval_metric': 'AUC'}
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
            'eval_metric': 'auc'}
        num_round = 100
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# ニューラルネットによるモデル
class Model1NN:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        batch_size = 128
        epochs = 10

        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(tr_x.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(tr_x, tr_y,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(va_x, va_y))
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)
        # pred = self.model.predict_proba(x).reshape(-1)
        pred = self.model.predict(x).reshape(-1)
        return pred


# 線形モデル
class Model2Linear:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(solver='lbfgs', C=1.0)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)[:, 1]
        return pred
    


import lightgbm as lgb



class Model1LightGBM:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 71
        }
        num_round = 100
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train)
        self.model = lgb.train(params, lgb_train, num_round, valid_sets=[lgb_train, lgb_eval])

    def predict(self, x):
        pred = self.model.predict(x, num_iteration=self.model.best_iteration)
        return pred

from catboost import CatBoostClassifier, Pool

class Model1CatBoost:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 71,
            'silent': True
        }
        train_pool = Pool(tr_x, label=tr_y)
        valid_pool = Pool(va_x, label=va_y)
        self.model = CatBoostClassifier(**params)
        self.model.fit(train_pool, eval_set=valid_pool, verbose_eval=False)

    def predict(self, x):
        pred = self.model.predict_proba(x)[:, 1]
        return pred



class Model1KNN:
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        # Set up the KNN classifier with K=16
        self.model = KNeighborsClassifier(n_neighbors=16)
        # Train the model
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        # Make predictions using the trained model
        pred = self.model.predict_proba(x)[:, 1]
        return pred
