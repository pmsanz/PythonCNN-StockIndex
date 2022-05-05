#
# Copyright (c) 2022. Pablo Sanz (pms.sanz@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 00:27:35 2021

@author: pmssa
"""

from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import compute_class_weight
from logger import Logger
from utils import reshape_as_image, get_db_directory
import matplotlib.pyplot as plt
import sqlalchemy
import time

plt.style.use('ggplot')

class DataLoader:
    def __init__(self, company_code = 'tBTCUSD', logger: Logger = None, mode_model = 'train'):
        
        #forced
        db_path = get_db_directory()
        dbname = 'tstream.db'
        engine_name = 'sqlite:///' + db_path + dbname
        self.engine = sqlalchemy.create_engine(engine_name)
        self.company_code = company_code
        self.mode_model = mode_model
        self.logger = logger
        self.start_col = 'open'
        self.end_col = 'eom_26'
        self.train_start = time.time()
        self.train_end = time.time()
        self.validation_start = time.time()
        self.validation_end = time.time()
        self.feat_idx = None
        self.reader = None
        self.df = pd.DataFrame()
        self.count_db = self.count_db()
        self.update_database_func()
        
        
    def update_database_func(self):
        
        self.load_data_db(1,20000)
        self.feat_idx = self.feature_selection()
        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.one_hot_enc.fit(self.df['labels'].values.reshape(-1, 1))
        self.valid_columns = self.df.columns.tolist()
        self.valid_columns.remove('labels')
        
    def count_db(self):
        
        sql_query = "SELECT COUNT(*) FROM {}".format(self.company_code)
        count = pd.read_sql_query(sql_query, self.engine)
        return int(count.iloc[:, 0])
    
    #load data to train
    def load_data_db(self, offset, limit = 1000):
                
        offset = offset * 1000
        sql_query = "SELECT * FROM {} ORDER BY timestamp LIMIT {} OFFSET {} ".format(self.company_code,str(limit),str(offset))
        df = pd.read_sql(sql_query, self.engine,parse_dates=["timestamp","timestamp_aux"])
                        
        df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
        df = df.set_index(df.timestamp)
        df = df.sort_index()
        
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        #correct columns
        
        self.df = df
        
        self.batch_start_date = self.df.iloc[0]['timestamp_aux']
        
        if offset + 1000 >= self.count_db:
            return True
        else:
            return False
        
        
        
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        


    #not modify
    def feature_selection(self):
       
        
        df_batch = self.df
        
        list_features = list(df_batch.loc[:, self.start_col:self.end_col].columns)
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
           
        x_train = mm_scaler.fit_transform(df_batch.loc[:, self.start_col:self.end_col].values)
        y_train = df_batch['labels'].values
        num_features = 225  # should be a perfect square
        topk = 350
       
        select_k_best = SelectKBest(f_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        common = list(set(selected_features_anova).intersection(selected_features_mic))
        
        print("common selected featues:" + str(len(common)) + ", " + str(common))
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                    num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        
        print(str(feat_idx))
        return feat_idx
    #not modify
    def df_by_date(self, start_date=None,end_date=None):
        
        if start_date == None:
            start_date = self.df.iloc[0]['timestamp_aux']
            
        if end_date == None:
            end_date = self.df.iloc[-1]['timestamp_aux']

       
        df_batch = self.df[(self.df['timestamp_aux'] >= start_date) & (self.df['timestamp_aux'] <= end_date)]
        return df_batch
    
    #not modify
    def get_data(self, start_date=None,end_date=None):
        
        df_batch = self.df_by_date(start_date,end_date)
        x = df_batch.loc[:, self.start_col:self.end_col].values
        x = x[:, self.feat_idx]
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x = mm_scaler.fit_transform(x)
        dim = int(np.sqrt(x.shape[1]))
        x = reshape_as_image(x, dim, dim)
        x = np.stack((x,) * 3, axis=-1)

        y = df_batch['labels'].values
        sample_weights = self.get_sample_weights(y)
        y = self.one_hot_enc.transform(y.reshape(-1, 1))

        return x, y, df_batch, sample_weights
    #not modify
    def get_sample_weights(self, y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.

        params:
            y: class labels as integers
        """
        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight(class_weight = "balanced",classes = np.unique(y),y=y)
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
            
        return sample_weights
      
    #nt modify
    def part_data(self,percent_test = 0.1):
        
        toal_lenght = len(self.df)
        units = int(toal_lenght * percent_test)
        start_test = toal_lenght - units
        start_date_train = self.df.iloc[0]['timestamp_aux']
        end_date_train = self.df.iloc[start_test]['timestamp_aux']
        start_date_test = self.df.iloc[start_test+1]['timestamp_aux']
        end_date_test = self.df.iloc[-1]['timestamp_aux']
   
        return start_date_train , end_date_train , start_date_test , end_date_test
     #nt modify   
    def get_data_new(self,count,percent_test = 0.1):
        
        is_last_batch = self.load_data_db(count)
                
        start_date_train , end_date_train , start_date_test , end_date_test = self.part_data(percent_test)
        
        x_train, y_train, df_batch_train, sample_weights = self.get_data(start_date_train,end_date_train)
        
        x_test, y_test, df_batch_test, sample_weights_test = self.get_data(start_date_test , end_date_test)
        
        x_train, x_cv, y_train, y_cv, sample_weights, _ = train_test_split(x_train, y_train, sample_weights,
                                                                           train_size=1 - percent_test,
                                                                           test_size=percent_test,
                                                                           random_state=2, shuffle=False
                                                                           )
            
        return x_train, y_train, x_cv, y_cv, x_test, y_test, df_batch_train, df_batch_test, sample_weights, is_last_batch
    
               
    #nt modify
    def convert_data(self,df,use_weights = True):
            
        df = df[self.valid_columns]
        x = df.loc[:, self.start_col:self.end_col].values
        x = x[:, self.feat_idx]
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x = mm_scaler.fit_transform(x)
        
        dim = int(np.sqrt(x.shape[1]))
        x = reshape_as_image(x, dim, dim)
        x = np.stack((x,) * 3, axis=-1)
        
        if self.mode_model == 'train':
            y = df['labels'].values
        
            if use_weights:
                sample_weights = self.get_sample_weights(y)
            else:
                y = y.astype(int)
                sample_weights = []
                
            y = self.one_hot_enc.transform(y.reshape(-1, 1))
        else:
            y = None
            sample_weights = []
        
        return x, y, df, sample_weights
       
     
    def get_max_columns_predicted(self,df_batch_test,predicted_values):
       
        predicted_values = pd.DataFrame(predicted_values)
        data = df_batch_test
        df = predicted_values.idxmax(axis=1)
        data = data.reset_index(drop=True)
        df = df.reset_index(drop=True)
        columns = { 'predicted': df.to_numpy(dtype='float64')}
        df_predicted_aux = pd.DataFrame(columns)
        data = pd.concat([data, df_predicted_aux], axis=1)
        data['timestamp'] = pd.to_datetime(data.timestamp_aux, unit='ms')
        data.set_index("timestamp", inplace = True)
        
        return data
   
    
