import numpy as np
import pandas as pd
import xlwings as xw


def jointdf(earlydata, laterdata, jointtime):
    jointtime_value = earlydata.loc[jointtime].values[0]
    laterdata_af = laterdata.loc[jointtime:, :]
    laterdata_af = 1 + laterdata_af.pct_change().fillna(jointtime_value - 1)
    laterdata_af = laterdata_af.cumprod()
    early_name = earlydata.columns[0]
    later_name = laterdata_af.columns[0]
    earlydata[later_name] = earlydata[early_name]
    earlydata = earlydata[[later_name]]
    earlydata = earlydata.iloc[:-1, :]
    full_data = pd.concat([earlydata, laterdata_af])
    full_data.index.set_names('Date', inplace=True)
    return full_data


def get_joint_data(ogdata, newdata, start='later'):
    og_end = ogdata.index[-1]
    new_start = newdata.index[0]
    if og_end < new_start:
        return pd.DataFrame(np.array([[0]]), columns=['数据错误'])
    else:
        if start == 'later':
            fulldf = jointdf(ogdata, newdata, og_end)
        else:
            fulldf = jointdf(ogdata, newdata, new_start)
    return fulldf


def joint_df():
    wb = xw.Book.caller()
    ogdata = wb.sheets['拼接'].range('B1:C6000').options(pd.DataFrame).value
    newdata = wb.sheets['拼接'].range('E1:F6000').options(pd.DataFrame).value
    ogdata = ogdata.dropna()
    newdata = newdata.dropna()
    ogdata.index = pd.to_datetime(ogdata.index)
    newdata.index = pd.to_datetime(newdata.index)
    full_data = get_joint_data(ogdata, newdata, start='later')
    name = wb.sheets['拼接'].range('H4').value
    wb.sheets['拼接'].range('I1').options(expand='table').value = full_data
    wb.sheets['拼接'].range('J1').options(expand='table').value = name
