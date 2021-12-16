import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from matplotlib import pyplot as plt
from datetime import timedelta
import datetime
import statsmodels.api as sm
from statsmodels import regression
from WindPy import w
import xlwings as xw

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.facecolor'] = 'snow'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def MY_judge_number_sign(x, set='positive'):
    if set == 'positive':
        if x > 0:
            y = 1
        else:
            y = 0
    else:
        if x < 0:
            y = 1
        else:
            y = 0

    return y


def MY_count_accelerated_number(given_series):
    container = []
    j = 0
    for i in given_series:
        if i == 1:
            j += 1
        else:
            j = 0
        container.append(j)
    return np.array(container)


def MY_count_accelerated_period_return(given_series):
    container = []
    j = 1
    for i in given_series:
        if i != 0:
            j = (1 + i) * j
        else:
            j = 1
        container.append(j)
    return container


def MY_get_excess_df(df):
    '''
    with benchmark in the last column
    '''
    df_col = df.columns.tolist()
    for i in df_col:
        df[i] = df[i] - df[df_col[-1]]
    df = df.iloc[:, :-1]
    return df


def MY_get_excess_nev1(df):
    '''
    with benchmark in the last column
    '''
    df = df / df.iloc[0]
    df1 = MY_get_excess_df(df) + 1
    return df1


def MY_get_excess_nev2(df):
    '''
    with benchmark in the last column
    '''
    df = df.pct_change()
    df1 = MY_get_excess_df(df) + 1
    df1 = df1.cumprod()
    df1.fillna(1, inplace=True)
    return df1


def MY_ret(df):
    """日/周收益率"""
    return (df / df.shift(1)).dropna() - 1


def MY_acc_ret(df):
    """累计收益率"""
    return df.values[-1] / df.values[0] - 1


def MY_acc_ret_df(df):
    """累计收益率序列"""
    return df / df.iloc[0] - 1


def MY_ann_ret(df):
    """年化收益率"""
    period = df.index[-1] - df.index[0]
    return pow(MY_acc_ret(df) + 1, 365 / period.days) - 1


def MY_ret_to_nev(df_ret):
    """根据收益算净值"""
    return (df_ret + 1).cumprod()


def MY_rec_week_ret(df, num):
    """
    计算近几周收益率
    num为周数
    """
    date = df.index[-1]
    df_week = df.tail(num + 1)
    return MY_acc_ret(df_week)


def MY_rec_interval_ret(df, period):
    '''
    :param df: dataframe
    :param period: one_month, three_months, six_months, one_year
    :return: a single number
    '''
    while True:
        target_value = 0
        try:
            target_value = df.loc[period].values[0]
            # print(target_value)
            break
        except:
            period = period - timedelta(days=1)
            # print(period)
            continue
    ret = df.values[-1][0] / target_value - 1
    return ret


def MY_year_return(df, year):
    """
    计算年度收益率
    """
    try:
        index2 = df.loc[str(year)].index[-1]
        date_start = df.index[0]
        try:
            index1 = df.loc[str(year - 1)].index[-1]
            index = max(index1, date_start)
        except:
            index = date_start
        df = df[index:index2]
        year_return = MY_acc_ret(df)
    except:
        year_return = ['/']
    return year_return


def MY_ann_para(fre="周"):
    """年化参数"""
    if fre == "日":
        ann_para = 250
    elif fre == "周":
        ann_para = 52
    else:
        ann_para = 12
    return ann_para


def MY_ann_vol(df, fre="周"):
    """年化波动率"""
    ann_coff = MY_ann_para(fre)
    return MY_ret(df).values.std() * np.sqrt(ann_coff)


def MY_ann_vol_down(df, fre="周"):
    """年化下行波动率"""
    ret_ = MY_ret(df)
    ret_down = (ret_[ret_ < 0]).dropna().values
    ann_coff = MY_ann_para(fre)
    return ret_down.std() * np.sqrt(ann_coff)


def MY_maxDD(df):
    """
    # 计算最大回撤率
    """
    df = df.values
    i = np.argmax((np.maximum.accumulate(df) - df) /
                  np.maximum.accumulate(df))  # 波谷
    if i == 0:
        return 0
    j = np.argmax(df[:i])
    maxdd = (df[j] - df[i]) / df[j]
    return maxdd


def MY_maxDD_sigle(df):
    """
    计算单期最大回撤
    """
    return -MY_ret(df).min().values


def MY_maxDD_back_period(df, backdays):
    '''
    :param df: dataframe
    :param backdays: 30, 90...
    :return: a single number
    '''
    back = df.index.tolist()[-1] - timedelta(days=backdays)
    period = back
    while True:
        backdate = 0
        try:
            test = df.loc[period]
            backdate = df.loc[period:]
            break
        except:
            period = period - timedelta(days=1)
            continue
    maxdd1 = MY_maxDD(backdate)

    return maxdd1


def MY_max_drawdown_trim(df):
    df1 = df.copy()
    max_drawdown_ratio = 0
    for e, i in enumerate(df1.iloc[:, 0].values):
        for f, j in enumerate(df1.iloc[:, 0].values):
            if f > e and float(j - i) / i < max_drawdown_ratio:
                max_drawdown_ratio = float(j - i) / i
                interval = [e, f]
    if interval[0] == interval[1]:
        print('No draw down interval')
        return ('/', '/')
    else:
        gap = df1.iloc[interval[0]] - df1.iloc[interval[1]]
        for i in range(interval[1], len(df1.index)):
            df1.iloc[i, 0] += gap
        return (df1.index[interval[0]].strftime('%Y-%m-%d'), df1.index[interval[1]].strftime('%Y-%m-%d'))


def MY_find_drawdown_interval(df):
    max_drawdown_ratio = 0
    for e, i in enumerate(df.iloc[:, 0].values):
        for f, j in enumerate(df.iloc[:, 0].values):
            if f > e and float(j - i) / i < max_drawdown_ratio:
                max_drawdown_ratio = float(j - i) / i
                interval = [e, f]
    if interval[0] == interval[1]:
        print('No draw down interval')
        return None, None, 0, df
    else:
        gap = df.iloc[interval[0]] - df.iloc[interval[1]]
        for i in range(interval[1], len(df.index)):
            df.iloc[i, 0] += gap
        df2 = df.drop(index=df.iloc[interval[0]:interval[1], ].index)
        return df.index[interval[0]].strftime('%Y-%m-%d'), df.index[interval[1]].strftime(
            '%Y-%m-%d'), max_drawdown_ratio, df2


def MY_make_drawdown_df(df, number, fre='周'):
    df_copy = df.copy()
    container = []
    for i in range(number):
        maxdd_start, maxdd_end, rate, df_copy = MY_find_drawdown_interval(df_copy)
        if rate == 0:
            container.append(['/', '/', '/', '/', '/', '/', '/'])
        else:
            if fre == '周':
                maxdd_lasting_periods = (
                        datetime.datetime.strptime(maxdd_end, '%Y-%m-%d') - datetime.datetime.strptime(maxdd_start,
                                                                                                       '%Y-%m-%d')).days
                maxdd_lasting_periods = round(maxdd_lasting_periods / 7)
                maxdd_lasting_periods = str(maxdd_lasting_periods) + fre
            else:
                maxdd_lasting_periods = (
                        datetime.datetime.strptime(maxdd_end, '%Y-%m-%d') - datetime.datetime.strptime(maxdd_start,
                                                                                                       '%Y-%m-%d')).days
                maxdd_lasting_periods = str(maxdd_lasting_periods) + fre

            maxdd_start_value = df.loc[datetime.datetime.strptime(maxdd_start, '%Y-%m-%d')].values[0]
            maxdd_end_value = df.loc[datetime.datetime.strptime(maxdd_end, '%Y-%m-%d')].values[0]
            maxdd_recover_df = df.loc[maxdd_start:df.index[-1], :]
            try:
                maxdd_recover_date = maxdd_recover_df[maxdd_recover_df[df.columns[0]] > maxdd_start_value].index[
                    0].strftime('%Y-%m-%d')
            except:
                maxdd_recover_date = '/'
            container.append(
                [format(rate, '.2%'), maxdd_start, maxdd_end, maxdd_start_value.round(4), maxdd_end_value.round(4),
                 maxdd_lasting_periods, maxdd_recover_date])
    drawdown_df = pd.DataFrame(index=["第1大回撤区间", "第2大回撤区间", "第3大回撤区间", "第4大回撤区间", "第5大回撤区间"],
                               columns=['历史回撤', '回撤波峰', '回撤波谷', '波峰净值', '波谷净值', '持续时间', '修复日期'],
                               data=np.array(container))

    return drawdown_df


def MY_Sharpe(df, rf=0.015, fre="周"):
    """夏普比率"""
    return (MY_ann_ret(df) - rf) / MY_ann_vol(df, fre)


def MY_Sharpe_excess(df1, df2, rf=0.015, fre="周"):
    '''
    :param df1: ret 超额净值序列
    :param df2: vol, maxdd 超额净值序列
    :return: a single number
    '''
    return (MY_ann_ret(df1) - rf) / MY_ann_vol(df2, fre)


def MY_Sortino(df, rf=0.015, fre="周"):
    """索提诺比率"""
    return (MY_ann_ret(df) - rf) / MY_ann_vol_down(df, fre)


def MY_Sortino_excess(df1, df2, rf=0.015, fre="周"):
    '''
    :param df1: ret 超额净值序列
    :param df2: vol, maxdd 超额净值序列
    :return: a single number
    '''
    return (MY_ann_ret(df1) - rf) / MY_ann_vol_down(df2, fre)


def MY_Calmar(df):
    """卡玛比率"""
    return MY_ann_ret(df) / MY_maxDD(df)


def MY_Calmar_excess(df1, df2):
    '''
    :param df1: ret 超额净值序列
    :param df2: vol, maxdd 超额净值序列
    :return: a single number
    '''
    return MY_ann_ret(df1) / MY_maxDD(df2)


def MY_win_loss(df):
    """
    计算盈亏比率
    """
    retdf = MY_ret(df)
    df = df.shift(1).dropna()
    df_ret = df * retdf
    return -(df_ret[df_ret > 0].sum() / df_ret[df_ret < 0].sum()).values


def MY_win_rate(df):
    """
    计算胜率
    """
    df_ret = MY_ret(df)
    return (df_ret[df_ret >= 0].count() / df_ret.count()).values


def MY_linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    # r_sq = model.rsquared
    return model.params[0], model.params[1]


def MY_self_linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    r_sq = model.rsquared
    return model.params[0], model.params[1], r_sq


def MY_get_none_matrix(row_num, col_num):
    a = []
    b = []
    for i in range(col_num):
        a.append(None)
    for i in range(row_num):
        b.append(a)
    return b


def MY_find_max_back_value_list(l):
    container = []
    index = 0
    for i in l:
        try:
            backmax = np.max(l[:index])
        except:
            backmax = 1
        index += 1
        container.append(backmax)

    return container