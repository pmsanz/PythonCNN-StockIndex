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

import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
import time
from predictCNN import PredictCNN
from datetime import datetime,date
from utils import get_db_directory
import os
plt.style.use('ggplot')


class BackTesting():
    def __init__(self, company_code, monto, gan_percent=0.002, num_trades_max = 10, interval = '5m', dbname = 'stream.db'):
        
        self.company_code = company_code
        self.initial_balance = monto
        self.actual_balance = monto
        self.stock_units = 0 
        self.trades = 0      
        self.position = 0
        self.perf = 0
        self.working = False
        self.gan_percent = gan_percent
        self.interval = interval
        self.origen = ""
        self.folder_name = str(time.time())
        self.archivo_log = 'log_' + str(date.today().strftime("%d_%m_%Y")) + '.txt'
        self.num_trades_max = num_trades_max
                
        db_path = get_db_directory()
        columns = {'index_aux':[],'position':[]}
        df_buy_sell = pd.DataFrame(columns)
        df_buy_sell.index_aux = df_buy_sell.index_aux.astype(int)
        df_buy_sell.position = df_buy_sell.position.astype(int)
        self.df_buy_sell = df_buy_sell
        engine_name = 'sqlite:///' + db_path + dbname
        self.engine = sqlalchemy.create_engine(engine_name)
        
        #loading methods
        self.create_folders()
        self.get_sql_data()
        
    def create_folders(self):
        os.mkdir(self.folder_name)
    
    #Drawn graphics
    def print_graphics(self, unit_index = 0):
        if self.trades > 0:
            self.update_position()
            fig = plt.figure(figsize=(12,9))
            ax1 = plt.subplot2grid((9,1), (1,0), rowspan=4, colspan=1)
            ax1.plot(self.data['close'], color='grey')      
            # Plotea Trades en Long
            ax1.plot(self.data[self.data['position'] == 1].index, 
                    self.data['close'][self.data['position'] == 1], '^', markersize = 10, color='green', label='Long')
            # Plotea Trades en Short
            ax1.plot(self.data[self.data['position'] == -1].index, 
                    self.data['close'][self.data['position'] == -1], 'v', markersize = 10, color='red', label='Short')
            
            plt.title(self.company_code +"\n IA Strategy", fontsize = 20)
            plt.legend(fontsize = 15)
            
            if self.trades >= self.num_trades_max:
                file_directory = self.folder_name + '/fig_' + str(self.trades) + '.png'
                plt.savefig(file_directory)
            
            plt.show(block=False)
            
            
    def update_position(self):
        for index in range(len(self.df_buy_sell)-1):
            new_df = pd.DataFrame({'position': [self.df_buy_sell['position'].iloc[index]]}, index=[self.df_buy_sell['index_aux'].iloc[index]])
            self.data.update(new_df)
            
    
    def return_performance(self):
        return self.perf
        
        
    def get_by_index(self, unit_index):
       date = str(self.data.index[unit_index].date())
       precio = round(self.data['close'].iloc[unit_index], 5)
       
       return date, precio
    
     # Buy method
    def make_long(self, unit_index, stock_units=None, monto=None):
        if monto:
            if monto == 'all':
                monto = self.actual_balance
                self.buy_instrument(unit_index, monto=monto) # Ir Long
         
        elif stock_units:
            self.buy_instrument(unit_index, stock_units= stock_units)
        
        elif self.position == -1: 
            self.buy_instrument(unit_index, stock_units= -self.stock_units) 
             

    # Sell method
    def make_short(self, unit_index, stock_units=None, monto=None):
        if monto:
            if monto == 'all':
                monto = self.actual_balance
                self.sell_instrument(unit_index, monto=monto) # Ir Short
        elif stock_units:
          self.sell_instrument(unit_index, stock_units = stock_units)
        elif  self.position == 1:
          self.sell_instrument(unit_index, stock_units= self.stock_units)

    def get_sql_data(self, df_aux = None):
            df = pd.read_sql(self.company_code, self.engine)
            
            df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
            df = df.set_index(df.timestamp)
            
            if 'timestamp' in df.columns:
                df = df.drop('timestamp', axis=1)
                
            if df_aux is not None:
                df['position'] = df_aux['position']
                df['position'] = df['position'].fillna(0)
                
            df = df.sort_index()
            self.data = df
            
    
    
    def save_log(self, msg):
        working_directory = self.folder_name +'/'+ self.archivo_log
        f = open(working_directory, "a")
        msg_aux = "{} - " + msg
        f.write(msg_aux.format(datetime.now()))
        f.write('\r')
        f.close()
       
    def get_last_index_DB(self,unit_index):
        
        if self.origen == 'realtime':
            len_db = len(self.data)
            len_new_db = len_db
            df_aux = self.data
            
            while len_db == len_new_db:
                self.get_sql_data(df_aux)
                len_new_db = len(self.data)
                
            print("Value loaded:" + str(self.data['close'].tail(1)))
            return len(self.data)-1
        
        else:
            if unit_index + 1 == len(self.data) - 1:
                self.working = False
            unit_index = unit_index + 1
            
            return unit_index+1
            
        

    def buy_instrument(self, unit_index, stock_units=None, monto=None):
        date, precio = self.get_by_index(unit_index)
        
        if monto is not None: 
          stock_units = monto / precio
        self.actual_balance -= stock_units * precio 
        self.stock_units += stock_units
        self.trades += 1
        print("{} | Buying: {}, {}, by: {}, trades counter: {}".format(date, stock_units, self.company_code, precio,self.trades))
        self.save_log("{} | Buying: {}, {}, by: {}, trades counter: {}".format(date, stock_units, self.company_code, precio,self.trades))
        
    def sell_instrument(self, unit_index, stock_units=None, monto=None):
        date, precio = self.get_by_index(unit_index)
       
        # if monto is not None: 
        stock_units = self.stock_units
        self.actual_balance += stock_units * precio 
        self.stock_units -= stock_units
        self.trades += 1
        print("{} | Selling: {}, {}, by: {}, trades counter: {}".format(date, stock_units, self.company_code, precio,self.trades))
        self.save_log("{} | Selling: {}, {}, by: {}, trades counter: {}".format(date, stock_units, self.company_code, precio,self.trades))
        
    def close_position(self, unit_index):
        date, precio = self.get_by_index(unit_index)
        print(100 * "-")
        print("{} | \-\-\-\-\-\ End of backtest /-/-/-/-/-/". format(date))
        self.save_log(100 * "-" + '\r' + "{} | \-\-\-\-\-\ FINAL DEL BACKTEST /-/-/-/-/-/". format(date))
        self.actual_balance += self.stock_units * precio 
        
        
        if self.position == 1:
            self.data.at[self.data.index[unit_index], 'position'] = -1
            self.trades += 1
            print("{} | Closing position : {} | by: {}".format(date, self.stock_units, precio))
            self.save_log("{} | Closing position : {} | by : {}, trades counter: {}".format(date, self.stock_units, precio,self.trades))
        
        
        self.stock_units = 0 
        
        self.perf = (self.actual_balance - self.initial_balance) / self.initial_balance * 100 
        #self.print_balance(unit_index)
        print("{} | net return(%): {}".format(date, round(self.perf, 2)))
        self.save_log("{} | net return(%): {}".format(date, round(self.perf, 2)))
        print("{} | trades counter: {}".format(date, self.trades))
        print(100 * "-") 
        self.save_log("{} | trades counter: {}".format(date, self.trades) +'\r' +100 * "-")

        
    
    def AI_strategy(self, rango=3, plot_graph=True):
        
        # Print data 
        stm = "VARIABLES: \r returns : {:f} \r".format(self.gan_percent)
        stm = stm + "Test Strategy: AI Strategy | {} ".format(self.company_code)
        print("-" * 100)
        print(stm)
        print("-" * 100)
        self.save_log("-" * 100)
        self.save_log(stm + '\r' + "-" * 100)
        
        # Reset Backtesting
        self.position = 0 # Neutral position
        self.trades   = 0
        self.actual_balance = self.initial_balance # No init capital
        self.working = True
                
        self.data['position'] = 0 
        unit_index = 0
        data_aux = pd.DataFrame()
        predict_cnn = PredictCNN(mode_model='use')
                
        while self.working:
                    
            unit_index = self.get_last_index_DB(unit_index)
            if self.origen == 'realtime':
               data_aux = self.data.tail(10)
            else:
                if unit_index < 96:
                    base = 0
                else:
                    base = unit_index - 96
                data_aux = self.data.iloc[base:unit_index+1]
                
            aux_pred = predict_cnn.predict_values(data_aux)
            #position = 1 : Sell
            #position = -1 : Buy
            print( aux_pred.iloc[-1]['timestamp_aux'].strftime("%d/%m/%Y, %H:%M:%S") + " Last value predicted: " + str(aux_pred.iloc[-1]['predicted']) + " price: " + str(aux_pred.iloc[-1]['close']))
            if self.position in [0,-1]: 
        # Buy
                if aux_pred.iloc[-1]['predicted'] == 1:
                    self.make_long(unit_index, monto = 'all') 
                    self.position = 1 
                    self.data.at[self.data.index[unit_index], 'position'] = self.position
            elif self.position in [1]: 
        # Sell
                if aux_pred.iloc[-1]['predicted'] == 0: 
                    self.make_short(unit_index, monto = 'all') 
                    self.position = -1 
                    self.data.at[self.data.index[unit_index], 'position'] = self.position
            if self.trades >= self.num_trades_max:
                self.working = False
                        
        if self.trades > 0:
            self.close_position(unit_index) # Cierra la position en la ultima unit_index
            self.df_buy_sell = self.df_buy_sell.append({'index_aux': unit_index+1,'position':1}, ignore_index=True) 
            self.print_graphics()
        else:
            print("There are not trades for this period of time. Try with another period.")
            print(100 * "-")
            self.save_log("There are not trades for this period of time. Try with another period." + '\r' + 100 * "-")
            
    def print_balance(self, unit_index):
        date, precio = self.get_by_index(unit_index)
        print("Initial balance: {}".format(round(self.initial_balance, 2)))
        self.save_log("Initial balance: {}".format(round(self.initial_balance, 2)))
        print("Actual balance: {}".format(round(self.actual_balance, 2)))
        self.save_log("{} | Actual balance: {}".format(date, round(self.actual_balance, 2)))
        
    def train_model(self):
        predict_cnn = PredictCNN(mode_model='train')
            

bkbase = BackTesting(company_code = "iBTCUSD", monto = 5000, 
                     gan_percent=0.0007,  
                     num_trades_max = 50, dbname = 'istream.db')
bkbase.train_model()
bkbase.AI_strategy()
print("Final returns : " + str(bkbase.return_performance()))
bkbase.save_log("Final returns: " + str(bkbase.return_performance()))




