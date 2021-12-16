import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import timedelta
import datetime
import statsmodels.api as sm
from statsmodels import regression
from WindPy import w
import xlwings as xw

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.facecolor'] = 'snow'
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
    interval = []
    max_drawdown_ratio = 0
    for e, i in enumerate(df1.iloc[:, 0].values):
        for f, j in enumerate(df1.iloc[:, 0].values):
            if f > e and float(j - i) / i <= max_drawdown_ratio:
                max_drawdown_ratio = float(j - i) / i
                interval = [e, f]
    gap = df1.iloc[interval[0]] - df1.iloc[interval[1]]
    for i in range(interval[1], len(df1.index)):
        df1.iloc[i, 0] += gap
    if max_drawdown_ratio != 0:
        return (df1.index[interval[0]].strftime('%Y-%m-%d'), df1.index[interval[1]].strftime('%Y-%m-%d'))
    else:
        return ('/', '/')


def MY_find_drawdown_interval(df):
    max_drawdown_ratio = 0
    interval = []
    for e, i in enumerate(df.iloc[:, 0].values):
        for f, j in enumerate(df.iloc[:, 0].values):
            if f > e and float(j - i) / i <= max_drawdown_ratio:
                # print((e, f), round(float(j - i) / i,4),round(max_drawdown_ratio,4))
                max_drawdown_ratio = float(j - i) / i
                interval = [e, f]
    # print(interval,max_drawdown_ratio)
    gap = df.iloc[interval[0]] - df.iloc[interval[1]]
    for i in range(interval[1], len(df.index)):
        df.iloc[i, 0] += gap
    df2 = df.drop(index=df.iloc[interval[0]:interval[1], ].index)
    if max_drawdown_ratio != 0:
        return df.index[interval[0]].strftime('%Y-%m-%d'), df.index[interval[1]].strftime(
            '%Y-%m-%d'), max_drawdown_ratio, df2
    else:
        # will not return date, need to modify!!!
        return df.index[interval[0]].strftime('%Y-%m-%d'), df.index[interval[1]].strftime(
            '%Y-%m-%d'), max_drawdown_ratio, df


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


###########################################################################################
# Main start:

def get_tradedates(type):
    now = datetime.datetime.now().date()
    if type == '日':
        df = w.tdays("2002-01-01", str(now), "", usedf=True)[1]
        df.columns = ['日期']
        df.set_index('日期', inplace=True)
    elif type == '周':
        df = w.tdays("2002-01-01", str(now), "Period=W", usedf=True)[1]
        df.columns = ['日期']
        df.set_index('日期', inplace=True)
    else:
        df = w.tdays("2002-01-01", str(now), "Period=M", usedf=True)[1]
        df.columns = ['日期']
        df.set_index('日期', inplace=True)
    return df


def get_benchmark(index, name):
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    # print(indeces['Name'].tolist())
    benchmark_df = w.wsd(index, "close", '2005-01-01', str(now), "", usedf=True)
    benchmark_df = benchmark_df[1]
    benchmark_df[name] = benchmark_df['CLOSE']
    benchmark_df = benchmark_df[[name]]
    return benchmark_df


def df_standard(df, dates_day, dates_week, dates_month, fre="周"):
    """
    数据频率标准化，fre可选日/周/月,默认取周
    """
    # df.index = pd.to_datetime(df.index)
    if fre == "日":
        dates = dates_day
    elif fre == "周":
        dates = dates_week
    else:
        dates = dates_month
    start = df.index[0]
    end = df.index[-1]
    end_value = df.iloc[-1, 0]
    df = dates_day.join(df, how='left').fillna(method='pad')
    df = dates.join(df, how='left')
    df = df.loc[start:end].fillna(method='pad').dropna(how="all")
    if fre in ['日', '周']:
        pass
    else:
        df.loc[end] = end_value

    return df


def df_standard_single(df, dates_day, dates_week, dates_month, fre="周"):
    """
    数据频率标准化，fre可选日/周/月,默认取周
    """

    if fre == "日":
        dates = dates_day
    elif fre == "周":
        dates = dates_week
    else:
        dates = dates_month
    start = df.index[0]
    end = df.index[-1]
    end_value = df.iloc[-1, 0]
    df = dates_day.join(df, how='left').fillna(method='pad')
    df = dates.join(df, how='left')
    df = df.loc[start:end].fillna(method='pad').dropna(how="all")
    if fre in ['日', '周']:
        pass
    else:
        df.loc[end] = end_value
    return df


def split_standard_dataframe(df, dates_day, dates_week, dates_month, fre="周", normalization=False):
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    if fre == '日':
        result_array = dates_day.copy()
    else:
        result_array = dates_week.copy()
    # name_list = df.iloc[:, [i % 2 == 1 for i in range(len(df.columns))]].columns.tolist()
    for i in range(int(len(df.columns) / 2)):
        sub_df = df.iloc[:, [2 * i, 2 * i + 1]]
        sub_df['date'] = sub_df[sub_df.columns[0]]
        sub_df1 = sub_df[['date', sub_df.columns[1]]]
        sub_df1.set_index(sub_df1.columns[0], inplace=True)
        sub_df1.dropna(inplace=True)
        try:
            sub_df2 = df_standard(sub_df1, dates_day, dates_week, dates_month, fre=fre)
            if normalization == True:
                sub_df2 = sub_df2 / sub_df2.iloc[0]
            else:
                pass
            result_array = result_array.join(sub_df2, how='left')
        except:
            pass
    result_array = result_array.dropna(how='all')
    return result_array


def to_percent(temp, position):
    return "%1.0f" % (100 * temp) + "%"


def get_outer_data(df, benchmark_df, excess=True):
    # DO NOT dropna
    df1 = df.copy()
    if excess == True:
        benchmark_df = benchmark_df.loc[df.index[0]:, :]
        benchmark_df1 = benchmark_df.apply(lambda x: MY_acc_ret_df(x)) + 1
        df1 = df1.join(benchmark_df1, how='left')
    else:
        pass
    full_outer_nev_df = df1
    full_ret = df1 / df1.shift(1) - 1
    outer_corr = full_ret.corr().round(4)
    return full_outer_nev_df, outer_corr


def get_inner_data(df, benchmark_df, excess=True):
    df1 = df.copy()
    if excess == True:
        df1 = df1.join(benchmark_df, how='left')
    else:
        pass
    full_df1 = df1.dropna()
    full_df1 = full_df1.apply(lambda x: MY_acc_ret_df(x)) + 1
    full_inner_nev_df = full_df1
    full_ret = full_df1.pct_change()
    inner_corr = full_ret.corr().round(4)
    return full_inner_nev_df, inner_corr


def get_excess_inner_data(df, benchmark_df):
    full_df = df
    full_df = pd.concat([full_df, benchmark_df], axis=1, join='inner')
    full_df1 = full_df.dropna()
    full_df2 = full_df1.copy()
    full_df1 = full_df1.apply(lambda x: MY_acc_ret_df(x))
    full_df1 = MY_get_excess_df(full_df1) + 1
    full_inner_excess_nev_df = full_df1

    full_ret = full_df2.pct_change()
    full_ret = MY_get_excess_df(full_ret)
    inner_excess_corr = full_ret.corr().round(4)
    return full_inner_excess_nev_df, inner_excess_corr


def get_interval_return(df):
    one_month = df.index.tolist()[-1] - timedelta(days=30)
    three_months = df.index.tolist()[-1] - timedelta(days=91)
    six_months = df.index.tolist()[-1] - timedelta(days=183)
    one_year = df.index.tolist()[-1] - timedelta(days=365)
    try:
        a = MY_rec_interval_ret(df, one_month).round(4)
    except:
        a = MY_rec_interval_ret(df, df.index[0]).round(4)
    try:
        b = MY_rec_interval_ret(df, three_months).round(4)
    except:
        b = MY_rec_interval_ret(df, df.index[0]).round(4)
    try:
        c = MY_rec_interval_ret(df, six_months).round(4)
    except:
        c = MY_rec_interval_ret(df, df.index[0]).round(4)
    try:
        d = MY_rec_interval_ret(df, one_year).round(4)
    except:
        d = MY_rec_interval_ret(df, df.index[0]).round(4)

    return a, b, c, d


def get_alpha_beta(df, benchmark_df, rf=0.015, fre="周"):
    benchmark_df = benchmark_df.loc[df.index, :]
    df_ret = df.pct_change().dropna()
    ben_ret = benchmark_df.pct_change().dropna()
    alpha, beta = MY_linreg(ben_ret.values, df_ret.values)
    annpara = MY_ann_para(fre=fre)
    alpha = alpha - (1 - beta) * rf / annpara
    return alpha, beta


def get_indicators(df, benchmark_df, fre="周", rf=0.015):
    df1 = df.copy()
    df1 = df1.dropna()
    starttime = df1.index[0].strftime('%Y-%m-%d')
    endtime = df1.index[-1].strftime('%Y-%m-%d')
    trading_period = int(len(df1.index.tolist()))
    alpha_beta = get_alpha_beta(df1, benchmark_df, rf=rf, fre=fre)
    beta = alpha_beta[1].round(4)
    alpha = alpha_beta[0].round(4)
    period_acc_ret = format(MY_acc_ret(df1)[0], '.2%')
    ann_ret = format(MY_ann_ret(df1)[0].round(4), '.2%')
    rencent = get_interval_return(df1)
    rec_one_mon = format(rencent[0], '.2%')
    rec_three_mons = format(rencent[1], '.2%')
    rec_six_mons = format(rencent[2], '.2%')
    rec_one_year = format(rencent[3], '.2%')
    if type(MY_year_return(df1, 2019)[0]) != str:
        ninet_ret = format(MY_year_return(df1, 2019)[0].round(4), '.2%')
    else:
        ninet_ret = MY_year_return(df1, 2019)[0]
    if type(MY_year_return(df1, 2020)[0]) != str:
        tw_ret = format(MY_year_return(df1, 2020)[0].round(4), '.2%')
    else:
        tw_ret = MY_year_return(df1, 2020)[0]
    if type(MY_year_return(df1, 2021)[0]) != str:
        twone_ret = format(MY_year_return(df1, 2021)[0].round(4), '.2%')
    else:
        twone_ret = MY_year_return(df1, 2021)[0]
    maxdd = format(-MY_maxDD(df1)[0].round(4), '.2%')
    period_maxdd = format(-MY_maxDD_sigle(df1)[0].round(4), '.2%')
    try:
        maxdd_three_mons = format(-MY_maxDD_back_period(df1, 90)[0].round(4), '.2%')
    except:
        maxdd_three_mons = format(-MY_maxDD_back_period(df1, (df1.index[-1] - df1.index[0]).days)[0].round(4), '.2%')

    try:
        maxdd_2021 = format(
            -MY_maxDD_back_period(df1, (df1.index[-1] - datetime.datetime(2021, 1, 1)).days)[0].round(4),
            '.2%')
    except:
        maxdd_2021 = format(-MY_maxDD_back_period(df1, (df1.index[-1] - df1.index[0]).days)[0].round(4), '.2%')

    maxdd_interval = MY_max_drawdown_trim(df1)
    maxdd_start = str(maxdd_interval[0])
    maxdd_end = str(maxdd_interval[1])
    ann_vol = format(MY_ann_vol(df1, fre=fre).round(4), '.2%')
    ann_vol_down = format(MY_ann_vol_down(df1, fre=fre).round(4), '.2%')
    sharpe = MY_Sharpe(df1, rf=rf, fre=fre)[0].round(2)
    calmar = MY_Calmar(df1)[0].round(2)
    sortino = MY_Sortino(df1, rf=rf, fre=fre)[0].round(2)
    win_loss = MY_win_loss(df1)[0].round(2)
    win_rate = format(MY_win_rate(df1)[0].round(4), '.2%')
    # print(df1.columns[0])
    output_df = pd.DataFrame(
        data=[starttime, endtime, str(fre), trading_period, beta, alpha, period_acc_ret,
              ann_ret, rec_one_mon, rec_three_mons, rec_six_mons, rec_one_year,
              ninet_ret, tw_ret, twone_ret, maxdd, maxdd_start, maxdd_end, period_maxdd, maxdd_three_mons,
              maxdd_2021, ann_vol, ann_vol_down, sharpe, calmar, sortino, win_loss, win_rate],
        index=['开始时间', '结束时间', '净值频率', '交易周期', 'Beta', 'Alpha', '区间累计收益率', '年化收益率',
               '近一个月收益率', '近三个月收益率', '近六个月收益率', '近一年收益率', '2019年累计收益率',
               '2020年累计收益率', '2021年累计收益率', '最大回撤', '最大回撤波峰', '最大回撤波谷', "单" + str(fre) + "最大回撤",
               '近三月最大回撤', '2021年最大回撤', '年化波动率', '年化下行波动率', '夏普比率', '卡玛比率',
               '索提诺比率', '盈亏比率', '周胜率'], columns=[df1.columns[0]])
    return output_df


def key_indicators_report(df, benchmark_df, fre="周", rf=0.015, excess=True):
    # full_df = pd.merge(df,benchmark_df,how='left')
    if excess == True:
        full_df = pd.concat([df, benchmark_df], axis=1, join='inner')
    else:
        full_df = df
    full_df1 = full_df.dropna()
    report_list = []
    for i in full_df1.columns:
        single_data = full_df1[[i]]
        report = get_indicators(single_data, benchmark_df, fre=fre, rf=rf)
        report_list.append(report)
    full_report = pd.concat(report_list, axis=1)
    full_report2 = full_report.copy()
    full_report2['产品名称'] = full_report2.index
    full_report2 = full_report2[['产品名称'] + full_report.columns.tolist()]
    return full_report2


def key_outer_indicators_report(df, benchmark_df, fre="周", rf=0.015, excess=True):
    full_df = df.copy()
    if excess == True:
        full_df1 = full_df.join(benchmark_df, how='left')
    else:
        full_df1 = full_df
    # print('\n\n', full_df1)
    report_list = []
    for i in full_df1.columns:
        single_data = full_df1[[i]]
        report = get_indicators(single_data, benchmark_df, fre=fre, rf=rf)
        report_list.append(report)
    full_report = pd.concat(report_list, axis=1)
    full_report2 = full_report.copy()
    full_report2['产品名称'] = full_report2.index
    full_report2 = full_report2[['产品名称'] + full_report.columns.tolist()]
    return full_report2


def get_excess_indicators(df, benchmark_df, fre="周", rf=0.015):
    df1 = pd.concat([df, benchmark_df], axis=1, join='inner')
    starttime = df1.index[0].strftime('%Y-%m-%d')
    endtime = df1.index[-1].strftime('%Y-%m-%d')
    # df_excess1_nev: ret (Difference approach)
    df_excess1_nev = MY_get_excess_nev1(df1)
    # df_excess2_nev: vol, maxdd (Product approach)
    df_excess2_nev = MY_get_excess_nev2(df1)
    trading_period = len(df.index.tolist())
    period_acc_ret = format(MY_acc_ret(df_excess1_nev)[0], '.2%')
    ann_ret = format(MY_ann_ret(df_excess1_nev)[0].round(4), '.2%')
    rencent = get_interval_return(df_excess1_nev)
    rec_one_mon = format(rencent[0], '.2%')
    rec_three_mons = format(rencent[1], '.2%')
    rec_six_mons = format(rencent[2], '.2%')
    rec_one_year = format(rencent[3], '.2%')
    maxdd = format(-MY_maxDD(df_excess2_nev)[0].round(4), '.2%')
    period_maxdd = format(-MY_maxDD_sigle(df_excess2_nev)[0].round(4), '.2%')
    try:
        maxdd_three_mons = format(-MY_maxDD_back_period(df_excess2_nev, 90)[0].round(4), '.2%')
    except:
        maxdd_three_mons = format(
            -MY_maxDD_back_period(df_excess2_nev, (df_excess2_nev.index[-1] - df_excess2_nev.index[0]).days)[0].round(
                4), '.2%')

    try:
        maxdd_2021 = format(
            -MY_maxDD_back_period(df_excess2_nev, (df_excess2_nev.index[-1] - datetime.datetime(2021, 1, 1)).days)[
                0].round(4),
            '.2%')
    except:
        maxdd_2021 = format(
            -MY_maxDD_back_period(df_excess2_nev, (df_excess2_nev.index[-1] - df_excess2_nev.index[0]).days)[0].round(
                4), '.2%')

    maxdd_interval = MY_max_drawdown_trim(df_excess2_nev)
    maxdd_start = str(maxdd_interval[0])
    maxdd_end = str(maxdd_interval[1])
    ann_vol = format(MY_ann_vol(df_excess2_nev, fre=fre).round(4), '.2%')
    ann_vol_down = format(MY_ann_vol_down(df_excess2_nev, fre=fre).round(4), '.2%')
    sharpe = MY_Sharpe_excess(df_excess1_nev, df_excess2_nev, rf=rf, fre=fre)[0].round(2)
    calmar = MY_Calmar_excess(df_excess1_nev, df_excess2_nev)[0].round(2)
    sortino = MY_Sortino_excess(df_excess1_nev, df_excess2_nev, rf=rf, fre=fre)[0].round(2)
    win_loss = MY_win_loss(df_excess1_nev)[0].round(2)
    win_rate = format(MY_win_rate(df_excess1_nev)[0].round(4), '.2%')
    # print(df.columns[0])
    output_df = pd.DataFrame(
        data=[starttime, endtime, str(fre), str(trading_period), period_acc_ret,
              ann_ret, rec_one_mon, rec_three_mons, rec_six_mons, rec_one_year,
              maxdd, maxdd_start, maxdd_end, period_maxdd, maxdd_three_mons,
              maxdd_2021, ann_vol, ann_vol_down, sharpe, calmar, sortino, win_loss, win_rate],
        index=['开始时间', '结束时间', '净值频率', '交易周期', '区间累计收益率', '年化收益率', '近一个月收益率', '近三个月收益率',
               '近六个月收益率', '近一年收益率', '最大回撤', '最大回撤波峰', '最大回撤波谷', "单" + str(fre) + "最大回撤",
               '近三月最大回撤', '2021年最大回撤', '年化波动率', '年化下行波动率', '夏普比率', '卡玛比率',
               '索提诺比率', '盈亏比率', '周胜率'], columns=[df.columns[0]])
    return output_df


def excess_indicators_report(df, benchmark_df, fre="周", rf=0.015):
    full_df = df
    full_df1 = full_df.dropna()
    report_list = []
    for i in full_df1.columns:
        single_data = full_df1[[i]]
        report = get_excess_indicators(single_data, benchmark_df, fre=fre, rf=rf)
        report_list.append(report)
    full_report = pd.concat(report_list, axis=1)
    full_report2 = full_report.copy()
    full_report2['产品名称'] = full_report2.index
    full_report2 = full_report2[['产品名称'] + full_report.columns.tolist()]
    return full_report2


def get_dynamic_drawdown(df):
    df0 = df.dropna()
    df1 = df0 / df0.iloc[0]
    namelist = df1.columns.tolist()
    df_ddlist = []
    for i in namelist:
        df_single = df1[[i]]
        df_single['max_back_values'] = np.array(MY_find_max_back_value_list(df_single[i].tolist()))
        df_single[i] = df_single[i] / df_single['max_back_values'] - 1
        df_single = df_single[[i]]
        df_ddlist.append(df_single)
    full_drawdown_df = pd.concat(df_ddlist, axis=1)
    return full_drawdown_df


def continous_plot_df(df, linestyle, dashed_last=True, marker=None):
    if dashed_last == True:
        df1 = df.iloc[:, :-1]
        df2 = df.iloc[:, -1:]
        names = df.columns
        fig = plt.figure(figsize=(10, 4))
        for i in names[:-1]:
            plt.plot(df1[[i]], label=i, linewidth=2, alpha=0.8, linestyle=linestyle, marker=marker)
        plt.plot(df2, label=names[-1], linewidth=2, alpha=0.8, linestyle='dashed', marker=marker)
    else:
        df1 = df
        names = df1.columns
        fig = plt.figure(figsize=(10, 4))
        for i in names:
            plt.plot(df1[[i]], label=i, linewidth=2, alpha=0.8, linestyle=linestyle, marker=marker)
    return fig


def make_my_compare_report(df, benchmark_data, sheetname, dates_day, dates_week, dates_month, workbook, fre="周",
                           rf=0.015, excess=True, write=False):
    benchmark_df = benchmark_data
    benchmark_df = df_standard_single(benchmark_df, dates_day, dates_week, dates_month, fre=fre)
    df1 = split_standard_dataframe(df, dates_day, dates_week, dates_month, fre=fre, normalization=True)
    full_outer_nev_df, outer_corr = get_outer_data(df1, benchmark_df, excess=excess)
    full_outer_nev_df = full_outer_nev_df - 1
    full_inner_nev_df, inner_corr = get_inner_data(df1, benchmark_df, excess=excess)
    full_inner_nev_df = full_inner_nev_df - 1
    # print('outer')
    # print(full_outer_nev_df)
    # print(outer_corr)
    # print('\n')
    # print('inner')
    # print(full_inner_nev_df)
    # print(inner_corr)
    # print('\n')
    key_outer_report = key_outer_indicators_report(df1, benchmark_df, fre=fre, rf=rf, excess=excess)
    # print('key_outer_report\n', key_outer_report)
    # print('\n')
    key_report = key_indicators_report(df1, benchmark_df, fre=fre, rf=rf, excess=excess)
    # print('key_report\n', key_report)
    # print('\n')
    ############################################# excess
    full_inner_excess_nev_df, inner_excess_corr = get_excess_inner_data(df1, benchmark_df)
    full_inner_excess_nev_df = full_inner_excess_nev_df - 1
    # print('\n')
    # print('inner_excess')
    # print(full_inner_excess_nev_df)
    # print(inner_excess_corr)
    # print('\n')
    excess_report = excess_indicators_report(df1, benchmark_df, fre=fre, rf=rf)
    # print('excess_report')
    # print(excess_report)
    full_dynamic_drawdown_df = get_dynamic_drawdown(df1)
    full_dynamic_drawdown_df.plot.area(stacked=False,alpha=0.5,figsize=(10, 4),colormap='Reds_r')
    plt.ylim(ymax=0)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('动态回撤')
    plt.tight_layout()
    plt.show()
    ###########################################################################################
    ax_outer = full_outer_nev_df.plot(figsize=(10, 4))
    # ax_outer = continous_plot_df(full_outer_nev_df,linestyle='-',dashed_last=True)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('净收益对比')
    plt.tight_layout()
    plt.show()
    fig_outer = ax_outer.get_figure()
    ax_inner = full_inner_nev_df.plot(figsize=(10, 4))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('净收益对比')
    plt.tight_layout()
    plt.show()
    fig_inner = ax_inner.get_figure()
    ax_inner_excess = full_inner_excess_nev_df.plot(figsize=(10, 4))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('超额收益对比')
    plt.tight_layout()
    plt.show()
    fig_inner_excess = ax_inner_excess.get_figure()
    if write == True:
        wb = workbook
        ####Clean data
        wb.sheets[sheetname].range('J1').options(expand='table').value = MY_get_none_matrix(row_num=150,
                                                                                            col_num=15)
        # fill data
        wb.sheets[sheetname].range('J14').value = '关键指标对比'
        wb.sheets[sheetname].range('J15').options(expand='table').value = key_outer_report.columns.tolist()
        wb.sheets[sheetname].range('J16').options(expand='table').value = key_outer_report.values
        wb.sheets[sheetname].range('J61').value = '同区间指标对比'
        wb.sheets[sheetname].range('J62').options(expand='table').value = key_report.columns.tolist()
        wb.sheets[sheetname].range('J63').options(expand='table').value = key_report.values
        wb.sheets[sheetname].range('J1').value = '相关性分析'
        wb.sheets[sheetname].range('J2').options(expand='table').value = outer_corr
        wb.sheets[sheetname].range('J48').value = '同区间相关性'
        wb.sheets[sheetname].range('J49').options(expand='table').value = inner_corr
        wb.sheets[sheetname].range('J95').value = '超额相关性'
        wb.sheets[sheetname].range('J96').options(expand='table').value = inner_excess_corr
        # wb.sheets[sheetname].range('AG1').options(expand='table').value = full_outer_nev_df
        # wb.sheets[sheetname].range('AU1').options(expand='table').value = full_inner_nev_df
        # wb.sheets[sheetname].range('BI1').options(expand='table').value = full_inner_excess_nev_df
        wb.sheets[sheetname].range('J107').value = '超额指标对比'
        wb.sheets[sheetname].range('J108').options(expand='table').value = excess_report.columns.tolist()
        wb.sheets[sheetname].range('J109').options(expand='table').value = excess_report.values
        wb.sheets[sheetname].pictures.add(fig_outer, name='Outer', update=True)
        wb.sheets[sheetname].pictures.add(fig_inner, name='Inner', update=True)
        wb.sheets[sheetname].pictures.add(fig_inner_excess, name='Inner_excess', update=True)
    else:
        pass
    # print('Done' + str(sheetname))


def mytest():
    w.start()
    w.isconnected()
    dates_day = get_tradedates(type='日')
    dates_week = get_tradedates(type='周')
    dates_month = get_tradedates(type='月')
    app = xw.App(visible=True, add_book=False)
    wb = app.books.open(r'compare_test2.xlsx')
    fre = wb.sheets['对比数据'].range('B2').value
    benchmark_code = wb.sheets['对比数据'].range('B4').value
    benchmark_name = wb.sheets['对比数据'].range('B3').value
    do_excess = wb.sheets['对比数据'].range('B6').value
    print(fre, benchmark_code, benchmark_name, do_excess)
    if do_excess == '是':
        excess = True
    else:
        excess = False
    ogdata = wb.sheets[0].range('D1:AA6000').options(pd.DataFrame).value
    ogdata = ogdata.reset_index(drop=True)
    ogdata = ogdata.dropna(how='all', axis=1)
    ogdata = ogdata.dropna(how='all', axis=0)
    print(ogdata)
    benchmark_data = get_benchmark(benchmark_code, benchmark_name)
    make_my_compare_report(ogdata, benchmark_data, '对比结果', dates_day, dates_week, dates_month, workbook=wb, fre=fre,
                           rf=0.015, excess=excess, write=True)

    wb.save()
    wb.close()
    app.quit()


mytest()
