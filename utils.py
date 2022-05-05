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

import re
import time
import numpy as np
import urllib.request
import shutil
import os
import pandas as pd
from PIL import Image
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from ta.others import *
from ta.volatility import *
from tqdm.auto import tqdm
from stockstats import StockDataFrame as sdf
from ta import *
from matplotlib import pyplot as plt


def seconds_to_minutes(seconds):
    return str(seconds // 60) + " minutes " + str(np.round(seconds % 60)) + " seconds"



def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")


def download_save(url, path_to_save, logger=None):
    if logger:
        logger.append_log("Starting download " + re.sub(r'apikey=[A-Za-z0-9]+&', 'apikey=my_api_key&', url))
    #else:
        ##print("Starting download " + re.sub(r'apikey=[A-Za-z0-9]+&', 'apikey=my_api_key&', url))
    urllib.request.urlretrieve(url, path_to_save)
    if logger:
        logger.append_log(path_to_save + " downloaded and saved")
    #else:
        ##print(path_to_save + " downloaded and saved")


def remove_dir(path):
    shutil.rmtree(path)
    ##print(path, "deleted")
    # os.rmdir(path)


def save_array_as_images(x, img_width, img_height, path, file_names):
    if os.path.exists(path):
        shutil.rmtree(path)
        ##print("deleted old files")

    os.makedirs(path)
    ##print("Image Directory created", path)
    x_temp = np.zeros((len(x), img_height, img_width))
    ##print("saving images...")
    # 
    for i in tqdm(range(x.shape[0])):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))
        img = Image.fromarray(x_temp[i], 'RGB')
        img.save(os.path.join(path, str(file_names[i]) + '.png'))

    ##print_time("Images saved at " + path, stime)
    return x_temp


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp


def show_images(rows, columns, path):
    # w = 15
    # h = 15
    fig = plt.figure(figsize=(15, 15))
    files = os.listdir(path)
    for i in range(1, columns * rows + 1):
        index = np.random.randint(len(files))
        img = np.asarray(Image.open(os.path.join(path, files[index])))
        fig.add_subplot(rows, columns, i)
        plt.title(files[i], fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(img)
    plt.show()


def dict_to_str(d):
    return str(d).replace("{", '').replace("}", '').replace("'", "").replace(' ', '')


def cleanup_file_path(path):
    return path.replace('\\', '/').replace(" ", "_").replace(':', '_')


def white_noise_check(tags_list, logger=None, *pd_series_args):
    if len(tags_list) != len(pd_series_args):
        raise Exception("Length of tags_list and series params different. Should be same.")
    for idx, s in enumerate(pd_series_args):
        # logger.append_log("1st, 2nd element {}, {}".format(s.iloc[0], s.iloc[1]))
        m = s.mean()
        std = s.std()
        logger.append_log("mean & std for {} is {}, {}".format(tags_list[idx], m, std))


def plot(y, title, output_path, x=None):
    fig = plt.figure(figsize=(10, 10))
    # x = x if x is not None else np.arange(len(y))
    plt.title(title)
    if x is not None:
        plt.plot(x, y, 'o-')
    else:
        plt.plot(y, 'o-')
        plt.savefig(output_path)


############### Technical indicators ########################


# not used
def get_RSI(df, col_name, intervals):
    """
    stockstats lib seems to use 'close' column by default so col_name
    not used here.
    This calculates non-smoothed RSI
    """
    df_ss = sdf.retype(df).copy()
    for i in intervals:
        df['rsi_' + str(i)] = df_ss['rsi_' + str(i)]

        del df['close_-1_s']
        del df['close_-1_d']
        del df['rs_' + str(i)]

        df['rsi_' + str(intervals[0])] = rsi(df['close'], i, fillna=True)
        
    ##print("RSI with stockstats done")


def get_RSI_smooth(df, col_name, intervals):
    """
    Momentum indicator
    As per https://www.investopedia.com/terms/r/rsi.asp
    RSI_1 = 100 - (100/ (1 + (avg gain% / avg loss%) ) )
    RSI_2 = 100 - (100/ (1 + (prev_avg_gain*13+avg gain% / prev_avg_loss*13 + avg loss%) ) )

    E.g. if period==6, first RSI starts from 7th index because difference of first row is NA
    http://cns.bu.edu/~gsc/CN710/fincast/Technical%20_indicators/Relative%20Strength%20Index%20(RSI).htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Verified!
    """

    ##print("Calculating RSI")
    
    prev_avg_gain = np.inf
    prev_avg_loss = np.inf
    rolling_count = 0

    def calculate_RSI(series, period):
        # nonlocal rolling_count
        nonlocal prev_avg_gain
        nonlocal prev_avg_loss
        nonlocal rolling_count


        # num_gains = (series >= 0).sum()
        # num_losses = (series < 0).sum()
        # sum_gains = series[series >= 0].sum()
        # sum_losses = np.abs(series[series < 0].sum())
        curr_gains = series.where(series >= 0, 0)  # replace 0 where series not > 0
        curr_losses = np.abs(series.where(series < 0, 0))
        avg_gain = curr_gains.sum() / period  # * 100
        avg_loss = curr_losses.sum() / period  # * 100
        rsi = -1

        if rolling_count == 0:
            # first RSI calculation
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
            # ##print(rolling_count,"rs1=",rs, rsi)
        else:
            # smoothed RSI
            # current gain and loss should be used, not avg_gain & avg_loss
            rsi = 100 - (100 / (1 + ((prev_avg_gain * (period - 1) + curr_gains.iloc[-1]) /
                                     (prev_avg_loss * (period - 1) + curr_losses.iloc[-1]))))
            # ##print(rolling_count,"rs2=",rs, rsi)

        # df['rsi_'+str(period)+'_own'][period + rolling_count] = rsi
        rolling_count = rolling_count + 1
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss
        return rsi

    diff = df[col_name].diff()[1:]  # skip na
    for period in tqdm(intervals):
        
        
        #df['rsi_' + str(period)] = np.nan
        # df['rsi_'+str(period)+'_own_1'] = np.nan
        rolling_count = 0
        res = diff.rolling(period).apply(calculate_RSI, args=(period,), raw=False)
        columns = {'rsi_' + str(period):res}
        dfrsi = pd.DataFrame(columns)
        #df['rsi_' + str(period)][1:] = res
        
        df = df.join(dfrsi, how='left')
        #df = pd.concat([df, dfrsi], axis=1)
        del dfrsi
    # df.drop(['diff'], axis = 1, inplace=True)
    ##print_time("Calculation of RSI Done", stime)
    return df


# not used: +1, ready to use
def get_IBR(df):
    return (df['close'] - df['low']) / (df['high'] - df['low'])


def get_williamR(df, col_name, intervals):
    """
    both libs gave same result
    Momentum indicator
    """
    
    ##print("Calculating WilliamR")
    df_ss = sdf.retype(df).copy()
    for i in tqdm(intervals):
        #df['wr_'+str(i)] = df_ss['wr_'+str(i)]
        columns = { 'wr_'+str(i) : df_ss['wr_'+str(i)] }
        df_ss_aux = pd.DataFrame(columns)
        #df['rsi_' + str(period)][1:] = res
        #df = pd.concat([df, df_ss_aux], axis=1)
        df = df.join(df_ss_aux, how='left')
        #del df_ss_aux
    del df_ss
    return df
        #df["wr_" + str(i)] = wr(df['high'], df['low'], df['close'], i, fillna=True)

    ##print_time("Calculation of WilliamR Done", stime)


def get_mfi(df, intervals):
    """
    momentum type indicator
    """

    
    ##print("Calculating MFI")
    for i in tqdm(intervals):
        #df['mfi_' + str(i)] = money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=i, fillna=True)
        columns = {'mfi_' + str(i) : money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=i, fillna=True)}
        dfmfi = pd.DataFrame(columns)
        #df['rsi_' + str(period)][1:] = res
        #df = pd.concat([df, dfmfi], axis=1)
        df = df.join(dfmfi, how='left')
        del dfmfi
    return df
    ##print_time("Calculation of MFI done", st
    ##print_time("Calculation of MFI done", stime)


def get_SMA(df, col_name, intervals):
    """
    Momentum indicator
    """
    
    ##print("Calculating SMA")
    df_ss = sdf.retype(df).copy()
    for i in tqdm(intervals):
        #df[col_name + '_sma_' + str(i)] = df_ss[col_name + '_' + str(i) + '_sma']
        
        columns = { col_name + '_sma_' + str(i) : df_ss[col_name + '_' + str(i) + '_sma'] }
        df_SMA_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_SMA_aux], axis=1)
        df = df.join(df_SMA_aux, how='left')
        del df_SMA_aux
    return df
        #del df[col_name + '_' + str(i) + '_sma_']

    ##print_time("Calculation of SMA Done", stime)


def get_EMA(df, col_name, intervals):
    """
    Needs validation
    Momentum indicator
    """
    
    ##print("Calculating EMA")
    df_ss = sdf.retype(df).copy()
    for i in tqdm(intervals):
        #df['ema_' + str(i)] = df_ss[col_name + '_' + str(i) + '_ema']
        columns = { 'ema_' + str(i) : df_ss[col_name + '_' + str(i) + '_ema'] }
        df_EMA_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_EMA_aux], axis=1)
        df = df.join(df_EMA_aux, how='left')
        del df_EMA_aux
    del df_ss
    return df

    ##print_time("Calculation of EMA Done", stime)


def get_WMA(df, col_name, intervals, hma_step=0):
    """
    Momentum indicator
    """
    
    if (hma_step == 0):
        # don't show progress for internal WMA calculation for HMA
        print("Calculating WMA")

    def wavg(rolling_prices, period):
        weights = pd.Series(range(1, period + 1))
        return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

    temp_col_count_dict = {}
    for i in tqdm(intervals, disable=(hma_step != 0)):
        res = df[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
        # print("interval {} has unique values {}".format(i, res.unique()))
        if hma_step == 0:
            df['wma_' + str(i)] = res
        elif hma_step == 1:
            if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
            else:
                temp_col_count_dict['hma_wma_' + str(i)] = 0
            # after halving the periods and rounding, there may be two intervals with same value e.g.
            # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
            df['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
        elif hma_step == 3:
            import re
            expr = r"^hma_[0-9]{1}"
            columns = list(df.columns)
            # print("searching", expr, "in", columns, "res=", list(filter(re.compile(expr).search, columns)))
            df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res

    return df

def get_HMA(df, col_name, intervals):
    import re
    
    ##print("Calculating HMA")
    expr = r"^wma_.*"

    if len(list(filter(re.compile(expr).search, list(df.columns)))) < 0:
        ##print("Need WMA first...")
        get_WMA(df, col_name, intervals)

    intervals_half = np.round([i / 2 for i in intervals]).astype(int)

    # step 1 = WMA for interval/2
    # this creates cols with prefix 'hma_wma_*'
    get_WMA(df, col_name, intervals_half, 1)
    # ##print("step 1 done", list(df.columns))

    # step 2 = step 1 - WMA
    columns = list(df.columns)
    expr = r"^hma_wma.*"
    hma_wma_cols = list(filter(re.compile(expr).search, columns))
    rest_cols = [x for x in columns if x not in hma_wma_cols]
    expr = r"^wma.*"
    wma_cols = list(filter(re.compile(expr).search, rest_cols))

    df[hma_wma_cols] = df[hma_wma_cols].sub(df[wma_cols].values,fill_value=0)  # .rename(index=str, columns={"close": "col1", "rsi_6": "col2"})
    # df[0:10].copy().reset_index(drop=True).merge(temp.reset_index(drop=True), left_index=True, right_index=True)

    # step 3 = WMA(step 2, interval = sqrt(n))
    intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
    for i, col in tqdm(enumerate(hma_wma_cols)):
        # ##print("step 3", col, intervals_sqrt[i])
        get_WMA(df, col, [intervals_sqrt[i]], 3)
    df.drop(columns=hma_wma_cols, inplace=True)
    ##print_time("Calculation of HMA Done", stime)
    return df

def get_TRIX(df, col_name, intervals):
    """
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    
    ##print("Calculating TRIX")
    #df_ss = sdf.retype(df).copy()
    for i in tqdm(intervals):
        # df['trix_'+str(i)] = df_ss['trix_'+str(i)+'_sma']
        #f['trix_' + str(i)] = trix(df['close'], i, fillna=True)
        columns = { 'trix_' + str(i) : trix(df['close'], i, fillna=True) }
        df_TRIX_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_TRIX_aux], axis=1)
        df = df.join(df_TRIX_aux, how='left')
        del df_TRIX_aux
    
    # df.drop(columns=['trix','trix_6_sma',])
    ##print_time("Calculation of TRIX Done", stime)
    return df

def get_DMI(df, col_name, intervals):
    """
    trend indicator
    TA gave same/wrong result
    """
    ##print("Calculating DMI")
    df_ss = sdf.retype(df).copy()
    for i in tqdm(intervals):
        columns = { 'dmi_' + str(i) : df_ss['adx_' + str(i) + '_ema'] }
        df_dmi_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_dmi_aux], axis=1)
        df = df.join(df_dmi_aux, how='left')
        
    drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema', 'pdm_14',
                    'close_-1_s', 'tr', 'tr_14_smma', 'atr_14']
   
    expr1 = r'dx_\d+_ema'
    expr2 = r'adx_\d+_ema'
    import re
    drop_columns.extend(list(filter(re.compile(expr1).search, list(df.columns)[9:])))
    drop_columns.extend(list(filter(re.compile(expr2).search, list(df.columns)[9:])))
    
    for name in drop_columns:
        if name in df.columns:
            df.drop(columns=name, inplace=True)
            
    del df_ss
    ##print_time("Calculation of DMI done", stime)
    return df

def get_CCI(df, col_name, intervals):
    
    
    for i in tqdm(intervals):
        columns = { 'cci_' + str(i) : cci(df['high'], df['low'], df['close'], i, fillna=True) }
        df_CCI_aux = pd.DataFrame(columns)
        df = df.join(df_CCI_aux, how='left')
    return df
    ##print_time("Calculation of CCI Done", stime)


def get_BB_MAV(df, col_name, intervals):
    """
    volitility indicator
    """
    ##print("Calculating Bollinger Band MAV")
    #df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        columns = { 'bb_' + str(i) : bollinger_mavg(df['close'], window=i, fillna=True) }
        df_BB_aux = pd.DataFrame(columns)
        df = df.join(df_BB_aux, how='left')
    return df
    ##print_time("Calculation of Bollinger Band MAV done", stime)


def get_CMO(df, col_name, intervals):
    """
    Chande Momentum Oscillator
    As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo

    CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
    range = +100 to -100

    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated

    return: None (adds the result in a column)
    """

    ##print("Calculating CMO")
    
    def calculate_CMO(series, period):
        # num_gains = (series >= 0).sum()
        # num_losses = (series < 0).sum()
        sum_gains = series[series >= 0].sum()
        sum_losses = np.abs(series[series < 0].sum())
        cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
        return np.round(cmo, 3)

    diff = df[col_name].diff()[1:]  # skip na
    for period in tqdm(intervals):
        
        res = diff.rolling(period).apply(calculate_CMO, args=(period,), raw=False)
        columns = { 'cmo_' + str(period) : res }
        df_CMO_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_CMO_aux], axis=1)
        df = df.join(df_CMO_aux, how='left')
        del df_CMO_aux
    return df
    ##print_time("Calculation of CMO Done", stime)


# not used. on close(12,16): +3, ready to use
def get_MACD(df):
    """
    Not used
    Same for both
    calculated for same 12 and 26 periods on close only!! Not different periods.
    creates colums macd, macds, macdh
    """
    ##print("Calculating MACD")
    df_ss = sdf.retype(df).copy()
    #df['macd'] = df_ss['macd']
    columns = { 'macd' : df_ss['macd'] }
    df_MACD_aux = pd.DataFrame(columns)
    #df = pd.concat([df, df_MACD_aux], axis=1)
    df = df.join(df_MACD_aux, how='left')
    del df_MACD_aux
    
    del df['macd_']
    del df['close_12_ema']
    del df['close_26_ema']
    ##print_time("Calculation of MACD done", stime)
    return df

# not implemented. period 12,26: +1, ready to use
def get_PPO(df, col_name, intervals):
    """
    As per https://www.investopedia.com/terms/p/ppo.asp

    uses EMA(12) and EMA(26) to calculate PPO value

    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated

    return: None (adds the result in a column)

    calculated for same 12 and 26 periods only!!
    """
    
    ##print("Calculating PPO")
    df_ss = sdf.retype(df).copy()
    
    #df['ema_' + str(12)] = df_ss[col_name + '_' + str(12) + '_ema']
    #del df['close_' + str(12) + '_ema']
    #df['ema_' + str(26)] = df_ss[col_name + '_' + str(26) + '_ema']
    #del df['close_' + str(26) + '_ema']
    #df['ppo'] = ((df['ema_12'] - df['ema_26']) / df['ema_26']) * 100

    columns = { 'ema_' + str(12) : df_ss[col_name + '_' + str(12) + '_ema'] , 'ema_' + str(26) : df_ss[col_name + '_' + str(26) + '_ema'], 'ppo': (df_ss[col_name + '_' + str(12) + '_ema']  - df_ss[col_name + '_' + str(26) + '_ema'] / df_ss[col_name + '_' + str(26) + '_ema']) * 100}
    df_PPO_aux = pd.DataFrame(columns)
    #df = pd.concat([df, df_PPO_aux], axis=1)
    df = df.join(df_PPO_aux, how='left')
    df = df.join(df_PPO_aux, how='left')
    del df_PPO_aux
    del df_ss
    # del df['ema_12']
    # del df['ema_26']
    return df
    ##print_time("Calculation of PPO Done", stime)


def get_ROC(df, col_name, intervals):
    """
    Momentum oscillator
    As per implement https://www.investopedia.com/terms/p/pricerateofchange.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    ROC = (close_price_n - close_price_(n-1) )/close_price_(n-1) * 100

    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated

    return: None (adds the result in a column)
    """
    
    ##print("Calculating ROC")

    def calculate_roc(series, period):
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

    for period in intervals:
        #df['roc_' + str(period)] = np.nan
        # for 12 day period, 13th day price - 1st day price
        res = df['close'].rolling(period + 1).apply(calculate_roc, args=(period,), raw=False)
        # ##print(len(df), len(df[period:]), len(res))
        #df['roc_' + str(period)] = res
        columns = { 'roc_' + str(period) : res }
        df_ROC_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_ROC_aux], axis=1)
        df = df.join(df_ROC_aux, how='left')
        del df_ROC_aux
    return df
    ##print_time("Calculation of ROC done", stime)


# not implemented, can't find
def get_PSI(df, col_name, intervals):
    """
    TODO implement
    """
    pass


def get_DPO(df, col_name, intervals):
    """
    Trend Oscillator type indicator
    """

    
    ##print("Calculating DPO")
    for i in tqdm(intervals):
        #df['dpo_' + str(i)] = dpo(df['close'], window=i)
        columns = { 'dpo_' + str(i) : dpo(df['close'], window=i) }
        df_DPO_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_DPO_aux], axis=1)
        df = df.join(df_DPO_aux, how='left')
        del df_DPO_aux
    ##print_time("Calculation of DPO done", stime)
    return df

def get_kst(df, col_name, intervals):
    """
    Trend Oscillator type indicator
    """

    
    ##print("Calculating KST")
    for i in tqdm(intervals):
        #df['kst_' + str(i)] = kst(df['close'], i)
        columns = { 'kst_' + str(i) : kst(df['close'], i) }
        df_kst_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_kst_aux], axis=1)
        df = df.join(df_kst_aux, how='left')
        del df_kst_aux
    ##print_time("Calculation of KST done", stime)
    return df

def get_CMF(df, col_name, intervals):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    
    ##print("Calculating CMF")
    for i in tqdm(intervals):
        #df['cmf_' + str(i)] = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], i, fillna=True)
        columns = { 'cmf_' + str(i) : chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], i, fillna=True) }
        df_CMF_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_CMF_aux], axis=1)
        df = df.join(df_CMF_aux, how='left')
        del df_CMF_aux
    ##print_time("Calculation of CMF done", stime)
    return df

def get_force_index(df, intervals):
    
    ##print("Calculating Force Index")
    for i in tqdm(intervals):
        #df['fi_' + str(i)] = force_index(df['close'], df['volume'], 5, fillna=True)
        columns = { 'fi_' + str(i) : force_index(df['close'], df['volume'], 5, fillna=True) }
        df_force_index_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_force_index_aux], axis=1)
        df = df.join(df_force_index_aux, how='left')
        del df_force_index_aux
    ##print_time("Calculation of Force Index done", stime)
    return df

def get_EOM(df, col_name, intervals):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    
    ##print("Calculating EOM")
    for i in tqdm(intervals):
        #df['eom_' + str(i)] = ease_of_movement(df['high'], df['low'], df['volume'], window=i, fillna=True)
        columns = { 'eom_' + str(i) : ease_of_movement(df['high'], df['low'], df['volume'], window=i, fillna=True) }
        #print("printingEom " + str(i))
        df_EOM_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_EOM_aux], axis=1)
        df = df.join(df_EOM_aux, how='left')
        #del df_EOM_aux
    return df
    ##print_time("Calculation of EOM done", stime)


# not used. +1
def get_volume_delta(df):
    
    ##print("Calculating volume delta")
    df_ss = sdf.retype(df).copy()
    df_ss['volume_delta']
    columns = { 'volume_delta' : df_ss['volume_delta'] }
    df_volume_delta_aux = pd.DataFrame(columns)
    #df = pd.concat([df, df_volume_delta_aux], axis=1)
    df = df.join(df_volume_delta_aux, how='left')
    del df_volume_delta_aux
    del df_ss
    ##print_time("Calculation of Volume Delta done", stime)
    return df

# not used. +2 for each interval kdjk and rsv
def get_kdjk_rsv(df, intervals):
    
    ##print("Calculating KDJK, RSV")
    df_ss = sdf.retype(df).copy()
    for i in tqdm(intervals):
        #df['kdjk_' + str(i)] = df_ss['kdjk_' + str(i)]
        columns = { 'kdjk_' + str(i) : df_ss['kdjk_' + str(i)] }
        df_kdjk_rsv_aux = pd.DataFrame(columns)
        #df = pd.concat([df, df_kdjk_rsv_aux], axis=1)
        df = df.join(df_kdjk_rsv_aux, how='left')
        del df_kdjk_rsv_aux
    del df_ss
    return df
    ##print_time("Calculation of EMA Done", stime)

def get_db_directory():
    from sys import platform
    db_path = ''
    if platform == "linux" or platform == "linux2":
        #linux
        root_path ='/root/environments/my_env3/src/'
        db_path = root_path + 'database/'
    else :
        #windows
        root_path = 'C:\\Repositorio\\portfolio\\Python - CNN StockIndex\\'
        db_path = root_path + 'database\\'
        
    return db_path