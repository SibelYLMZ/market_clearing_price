import pandas as pd
import holidays
import numpy as np
import data




def train_test_split(data_epias, data_metalogica):
    
    df_epias = data_epias.copy()

    data_metalogica = data_metalogica.loc[:,~data_metalogica.columns.duplicated()].copy()
    df_metalogica = data_metalogica[['Timestamp start (Asia/Istanbul)',
                                    'Hydro generation Run of River forecast Meteologica Turkey (GW)',
                                    'Hydro generation Conventional forecast Meteologica Turkey (GW)',
                                    'Wind power forecast Meteologica Turkey (GW)',
                                    'Photovoltaic Licensed forecast Meteologica Turkey (GW)',
                                    'Photovoltaic Unlicensed forecast Meteologica Turkey (GW)',
                                    'Power demand average forecast ECMWF ENS Turkey (GW)']]
    # df_metalogica = df_metalogica.drop(df_metalogica.columns[[1,2,3]], axis = 1)
    
    exos = ['ds','river', 'dam', 'wind', 'sun1', 'sun2', 'alis']
    df_metalogica = df_metalogica.rename(columns=dict(zip(df_metalogica.columns,exos)))
    df_metalogica = df_metalogica.iloc[:96]
    df_metalogica['ds'] = pd.to_datetime(df_metalogica['ds'])
    df_metalogica['sun'] = df_metalogica['sun1'] + df_metalogica['sun2']
    df_metalogica = df_metalogica.drop(['sun1', 'sun2'], axis = 1)
    df_metalogica.loc[:,["river", "dam", "wind", "alis", "sun"]] = df_metalogica.loc[:,["river", "dam", "wind", "alis", "sun"]].multiply(1000, axis="index")
    
    holiday = pd.DataFrame(columns=['tarih', 'holiday'])
    for date, name in sorted(holidays.Turkey(years=[2022, 2023,2024]).items()):
        holiday.loc[len(holiday)] = [date, name]
    holiday['tarih'] = pd.to_datetime(holiday['tarih'], format='%Y-%m-%d', errors='ignore')

    # df_epias_start = str(df_epias['ds'].iloc[0])
    # df_epias_end = str(df_epias['ds'].iloc[-1])
    df_epias_ds = df_epias['ds']
    df_epias['ds'] = pd.to_datetime(df_epias['ds']).dt.date
    df_epias = df_epias.set_index('ds').join(holiday.set_index('tarih')).reset_index(names = 'ds')
    # df_epias['ds'] = pd.date_range(start = df_epias_start, end = df_epias_end, freq = 'H')
    df_epias['ds'] = df_epias_ds

    # df_metalogica_start = str(df_metalogica['ds'].iloc[0])
    # df_metalogica_end = str(df_metalogica['ds'].iloc[-1])
    df_metalogica_ds = df_metalogica['ds']
    df_metalogica['ds'] = pd.to_datetime(df_metalogica['ds']).dt.date
    df_metalogica = df_metalogica.set_index('ds').join(holiday.set_index('tarih')).reset_index(names = 'ds')
    # df_metalogica['ds'] = pd.date_range(start = df_metalogica_start, end = df_metalogica_end, freq = 'H')
    df_metalogica['ds'] = df_metalogica_ds

    df_epias['holiday'] = df_epias['holiday'].astype(str)
    df_epias.loc[df_epias['holiday'] != 'nan', 'holiday'] = 1
    df_epias.loc[df_epias['holiday'] == 'nan', 'holiday'] = 0
    n2 = df_epias[df_epias['ds'].isin(pd.to_datetime(df_epias['ds'] + pd.DateOffset(day=2)))].index[0]
    df_epias['is_holiday_lead_2'] = df_epias['holiday'].shift(-n2)
    df_epias.loc[df_epias['is_holiday_lead_2'].isnull(), 'is_holiday_lead_2' ] = 0
    df_epias['holiday'] = df_epias['holiday'].astype(int)
    df_epias['is_holiday_lead_2'] = df_epias['is_holiday_lead_2'].astype(int)

    df_metalogica['holiday'] = df_metalogica['holiday'].astype(str)
    df_metalogica.loc[df_metalogica['holiday'] != 'nan', 'holiday'] = 1
    df_metalogica.loc[df_metalogica['holiday'] == 'nan', 'holiday'] = 0
    # n2 = df_metalogica[df_metalogica['ds'].isin(pd.to_datetime(df_metalogica['ds'] + pd.DateOffset(day=2)))].index[0]
    df_metalogica['is_holiday_lead_2'] = df_metalogica['holiday'].shift(-n2)
    df_metalogica.loc[df_metalogica['is_holiday_lead_2'].isnull(), 'is_holiday_lead_2' ] = 0
    df_metalogica['holiday'] = df_metalogica['holiday'].astype(int)
    df_metalogica['is_holiday_lead_2'] = df_metalogica['is_holiday_lead_2'].astype(int)

    df_epias['dayofmonth'] = df_epias['ds'].dt.day
    df_epias['dayofweek'] = df_epias['ds'].dt.dayofweek
    df_epias['quarter'] = df_epias['ds'].dt.quarter
    df_epias['month'] = df_epias['ds'].dt.month
    df_epias['year'] = df_epias['ds'].dt.year
    df_epias['dayofyear'] = df_epias['ds'].dt.dayofyear
    df_epias['weekofyear'] = df_epias['ds'].dt.isocalendar().week
    df_epias['hour'] =  df_epias['ds'].dt.hour
    df_epias['weekofyear'] = df_epias['weekofyear'].astype(int)

    df_metalogica['dayofmonth'] = df_metalogica['ds'].dt.day
    df_metalogica['dayofweek'] = df_metalogica['ds'].dt.dayofweek
    df_metalogica['quarter'] = df_metalogica['ds'].dt.quarter
    df_metalogica['month'] = df_metalogica['ds'].dt.month
    df_metalogica['year'] = df_metalogica['ds'].dt.year
    df_metalogica['dayofyear'] = df_metalogica['ds'].dt.dayofyear
    df_metalogica['weekofyear'] = df_metalogica['ds'].dt.isocalendar().week
    df_metalogica['hour'] =  df_metalogica['ds'].dt.hour
    df_metalogica['weekofyear'] = df_metalogica['weekofyear'].astype(int)

    df_epias['mondayRise'] = np.where((df_epias['dayofweek'] == 0) & ((df_epias['hour'] == 8) | (df_epias['hour'] == 9) ) | ((df_epias['hour'] >= 17) & (df_epias['hour'] <=21)), 1, 0)
    df_epias['tuesdayRise'] = np.where((df_epias['dayofweek'] == 1) & ((df_epias['hour'] == 8) | (df_epias['hour'] == 9) ) | ((df_epias['hour'] >= 17) & (df_epias['hour'] <=21)), 1, 0)
    df_epias['wednesdayRise'] = np.where((df_epias['dayofweek'] == 2) & ((df_epias['hour'] == 8) | (df_epias['hour'] == 9) ) | ((df_epias['hour'] >= 17) & (df_epias['hour'] <=21)), 1, 0)
    df_epias['thursdayRise'] = np.where((df_epias['dayofweek'] == 3) & ((df_epias['hour'] == 8) | (df_epias['hour'] == 9) ) | ((df_epias['hour'] >= 17) & (df_epias['hour'] <=21)), 1, 0)
    df_epias['fridayRise'] = np.where((df_epias['dayofweek'] == 4) & ((df_epias['hour'] == 8) | (df_epias['hour'] == 9) ) | ((df_epias['hour'] >= 17) & (df_epias['hour'] <=21)), 1, 0)
    df_epias['saturdayRise'] = np.where((df_epias['dayofweek'] == 5) & ((df_epias['hour'] >= 18) & (df_epias['hour'] <=21)), 1, 0)
    df_epias['sundayRise'] = np.where((df_epias['dayofweek'] == 6) & (df_epias['hour'] == 19), 1, 0)
    df_epias['highestDay'] = np.where((df_epias['dayofweek'] == 3),  1, 0)

    df_metalogica['mondayRise'] = np.where((df_metalogica['dayofweek'] == 0) & ((df_metalogica['hour'] == 8) | (df_metalogica['hour'] == 9) ) | ((df_metalogica['hour'] >= 17) & (df_metalogica['hour'] <=21)), 1, 0)
    df_metalogica['tuesdayRise'] = np.where((df_metalogica['dayofweek'] == 1) & ((df_metalogica['hour'] == 8) | (df_metalogica['hour'] == 9) ) | ((df_metalogica['hour'] >= 17) & (df_metalogica['hour'] <=21)), 1, 0)
    df_metalogica['wednesdayRise'] = np.where((df_metalogica['dayofweek'] == 2) & ((df_metalogica['hour'] == 8) | (df_metalogica['hour'] == 9) ) | ((df_metalogica['hour'] >= 17) & (df_metalogica['hour'] <=21)), 1, 0)
    df_metalogica['thursdayRise'] = np.where((df_metalogica['dayofweek'] == 3) & ((df_metalogica['hour'] == 8) | (df_metalogica['hour'] == 9) ) | ((df_metalogica['hour'] >= 17) & (df_metalogica['hour'] <=21)), 1, 0)
    df_metalogica['fridayRise'] = np.where((df_metalogica['dayofweek'] == 4) & ((df_metalogica['hour'] == 8) | (df_metalogica['hour'] == 9) ) | ((df_metalogica['hour'] >= 17) & (df_metalogica['hour'] <=21)), 1, 0)
    df_metalogica['saturdayRise'] = np.where((df_metalogica['dayofweek'] == 5) & ((df_metalogica['hour'] >= 18) & (df_metalogica['hour'] <=21)), 1, 0)
    df_metalogica['sundayRise'] = np.where((df_metalogica['dayofweek'] == 6) & (df_metalogica['hour'] == 19), 1, 0)
    df_metalogica['highestDay'] = np.where((df_metalogica['dayofweek'] == 3),  1, 0)

    return df_epias, df_metalogica









