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
    '''
    ONLY ONE drawdown interval,use find_drawdown_interval to find more
    '''
    df1 = df.copy()
    interval = []
    max_drawdown_ratio = 0
    for e, i in enumerate(df1.iloc[:, 0].values):
        for f, j in enumerate(df1.iloc[:, 0].values):
            if f > e and float(j - i) / i <= max_drawdown_ratio:
                max_drawdown_ratio = float(j - i) / i
                interval = [e, f]
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
                max_drawdown_ratio = float((j - i) / i)
                interval = [e, f]
    if max_drawdown_ratio != 0:
        gap_ratio = df.iloc[interval[0]].values[0] / df.iloc[interval[1]].values[0]
        for i in range(interval[1], len(df.index)):
            df.iloc[i, 0] = df.iloc[i, 0] * gap_ratio
        df2 = df.drop(index=df.iloc[interval[0]:interval[1], ].index)
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
    alpha, beta, r_sq = MY_self_linreg(ben_ret.values, df_ret.values)
    annpara = MY_ann_para(fre=fre)
    alpha = alpha - (1 - beta) * rf / annpara
    return alpha, beta, r_sq


def get_adjacent_indicators_win_loss(df):
    '''
    df: single column df with datetime_index
    '''
    # print(df)
    ret = df.pct_change().dropna()
    win_loss_period_df = ret.copy()
    ret = ret.sort_values(by=df.columns[0], ascending=False)
    first_win_value = format(ret.iloc[0].values[0].round(4), '.2%')
    first_win_date = ret.index[0].strftime('%Y-%m-%d')
    first_loss_value = format(ret.iloc[-1].values[0].round(4), '.2%')
    first_loss_date = ret.index[-1].strftime('%Y-%m-%d')
    second_win_value = format(ret.iloc[1].values[0].round(4), '.2%')
    second_win_date = ret.index[1].strftime('%Y-%m-%d')
    second_loss_value = format(ret.iloc[-2].values[0].round(4), '.2%')
    second_loss_date = ret.index[-2].strftime('%Y-%m-%d')
    container = [first_win_date, first_win_value, second_win_date, second_win_value, first_loss_date, first_loss_value,
                 second_loss_date, second_loss_value]
    win_loss_period_df['win'] = win_loss_period_df[df.columns[0]].apply(
        lambda x: MY_judge_number_sign(x, set='positive'))
    win_loss_period_df['loss'] = win_loss_period_df[df.columns[0]].apply(
        lambda x: MY_judge_number_sign(x, set='negative'))
    win_loss_period_df['acc_win'] = MY_count_accelerated_number(win_loss_period_df['win'])
    win_loss_period_df['acc_loss'] = MY_count_accelerated_number(win_loss_period_df['loss'])
    max_acc_win_period = win_loss_period_df['acc_win'].max()
    max_acc_loss_period = win_loss_period_df['acc_loss'].max()
    win_loss_period_df['acc_win_ratio'] = MY_count_accelerated_period_return(
        win_loss_period_df[df.columns[0]] * win_loss_period_df['win'])
    win_loss_period_df['acc_loss_ratio'] = MY_count_accelerated_period_return(
        win_loss_period_df[df.columns[0]] * win_loss_period_df['loss'])
    max_acc_win_value = win_loss_period_df['acc_win_ratio'].max() - 1
    max_acc_loss_value = win_loss_period_df['acc_loss_ratio'].min() - 1
    container2 = [max_acc_win_period, max_acc_win_value, max_acc_loss_period, max_acc_loss_value]

    return container, container2


def get_indicators(df, benchmark_df, benchmark_name, fre="周", rf=0.015):
    trading_period = int(len(df.index.tolist()))
    latest_nev = df.iloc[-1, 0]
    alpha_beta_r = get_alpha_beta(df, benchmark_df, rf=rf, fre=fre)
    beta = alpha_beta_r[1].round(4)
    alpha = alpha_beta_r[0].round(4)
    r_sq = alpha_beta_r[2].round(4)
    period_acc_ret = format(MY_acc_ret(df)[0], '.2%')
    ann_ret = format(MY_ann_ret(df)[0].round(4), '.2%')
    rencent = get_interval_return(df)
    rec_one_mon = format(rencent[0], '.2%')
    rec_three_mons = format(rencent[1], '.2%')
    rec_six_mons = format(rencent[2], '.2%')
    rec_one_year = format(rencent[3], '.2%')
    if type(MY_year_return(df, 2018)[0]) != str:
        eighte_ret = format(MY_year_return(df, 2018)[0].round(4), '.2%')
    else:
        eighte_ret = MY_year_return(df, 2018)[0]
    if type(MY_year_return(df, 2019)[0]) != str:
        ninet_ret = format(MY_year_return(df, 2019)[0].round(4), '.2%')
    else:
        ninet_ret = MY_year_return(df, 2019)[0]
    if type(MY_year_return(df, 2020)[0]) != str:
        tw_ret = format(MY_year_return(df, 2020)[0].round(4), '.2%')
    else:
        tw_ret = MY_year_return(df, 2020)[0]
    if type(MY_year_return(df, 2021)[0]) != str:
        twone_ret = format(MY_year_return(df, 2021)[0].round(4), '.2%')
    else:
        twone_ret = MY_year_return(df, 2021)[0]

    maxdd = format(-MY_maxDD(df)[0].round(4), '.2%')
    # period_maxdd = format(-MY_maxDD_sigle(df)[0].round(4), '.2%')
    try:
        maxdd_three_mons = format(-MY_maxDD_back_period(df, 90)[0].round(4), '.2%')
    except:
        maxdd_three_mons = format(-MY_maxDD_back_period(df, (df.index[-1] - df.index[0]).days)[0].round(4), '.2%')

    try:
        maxdd_six_mons = format(-MY_maxDD_back_period(df, 180)[0].round(4), '.2%')
    except:
        maxdd_six_mons = format(-MY_maxDD_back_period(df, (df.index[-1] - df.index[0]).days)[0].round(4), '.2%')

    try:
        maxdd_2021 = format(-MY_maxDD_back_period(df, (df.index[-1] - datetime.datetime(2021, 1, 1)).days)[0].round(4), '.2%')
    except:
        maxdd_2021 = format(-MY_maxDD_back_period(df, (df.index[-1] - df.index[0]).days)[0].round(4), '.2%')

    adjacent_indicators_win_loss_list, acc_win_loss_list = get_adjacent_indicators_win_loss(df)
    first_win_date = adjacent_indicators_win_loss_list[0]
    first_win_value = adjacent_indicators_win_loss_list[1]
    second_win_date = adjacent_indicators_win_loss_list[2]
    second_win_value = adjacent_indicators_win_loss_list[3]
    first_loss_date = adjacent_indicators_win_loss_list[4]
    first_loss_value = adjacent_indicators_win_loss_list[5]
    second_loss_date = adjacent_indicators_win_loss_list[6]
    second_loss_value = adjacent_indicators_win_loss_list[7]
    max_acc_win_period, max_acc_win_value = acc_win_loss_list[0], format(acc_win_loss_list[1], '.2%')
    max_acc_loss_period, max_acc_loss_value = acc_win_loss_list[2], format(acc_win_loss_list[3], '.2%')
    ann_vol = format(MY_ann_vol(df, fre=fre).round(4), '.2%')
    ann_vol_down = format(MY_ann_vol_down(df, fre=fre).round(4), '.2%')
    sharpe = MY_Sharpe(df, rf=rf, fre=fre)[0].round(2)
    calmar = MY_Calmar(df)[0].round(2)
    sortino = MY_Sortino(df, rf=rf, fre=fre)[0].round(2)
    df_back_1yr = df.loc[(df.index[-1] - timedelta(days=365)):df.index[-1], :]
    try:
        sharpe_back_1yr = MY_Sharpe(df_back_1yr, rf=rf, fre=fre)[0].round(2)
    except:
        sharpe_back_1yr = '/'
    win_loss = MY_win_loss(df)[0].round(2)
    win_rate = format(MY_win_rate(df)[0].round(4), '.2%')
    # print(df.columns[0])
    key_df_dict = {'name1': ['交易周期', '累计盈利', '回归基准', 'Alpha', '近一个月收益率', '近六个月收益率', '2018年累计收益率',
                             '2020年累计收益率', "第一大单" + str(fre) + "盈利日期", "第一大单" + str(fre) + "亏损日期",
                             "第二大单" + str(fre) + "盈利日期", "第二大单" + str(fre) + "亏损日期", '最大连续盈利时间长度',
                             '最大连续亏损时间长度', "历史最大回撤", '近六月最大回撤', '年化波动率', '近一年夏普比率', '卡玛比率',
                             str(fre) + '胜率'],
                   'value1': [str(trading_period) + str(fre), period_acc_ret, benchmark_name, alpha, rec_one_mon,
                              rec_six_mons, eighte_ret,
                              tw_ret, first_win_date, first_loss_date, second_win_date,
                              second_loss_date, max_acc_win_period, max_acc_loss_period, maxdd, maxdd_six_mons,
                              ann_vol, sharpe_back_1yr, calmar, win_rate],
                   'name2': ['最新净值', '年化收益率', 'Beta', 'R平方', '近三个月收益率', '近一年收益率', '2019年累计收益率',
                             '2021年累计收益率', "第一大单" + str(fre) + "盈利率", "第二大单" + str(fre) + "盈利率",
                             "第一大单" + str(fre) + "亏损率", "第二大单" + str(fre) + "亏损率", '对应连续盈利占比',
                             '对应连续亏损占比', '近三月最大回撤', '2021年最大回撤', '年化下行波动率', '统计区间夏普比率',
                             '索提诺比率', '盈亏比率'],
                   'value2': [latest_nev, ann_ret, beta, r_sq, rec_three_mons, rec_one_year, ninet_ret, twone_ret,
                              first_win_value, second_win_value, first_loss_value, second_loss_value, max_acc_win_value,
                              max_acc_loss_value, maxdd_three_mons, maxdd_2021, ann_vol_down, sharpe, sortino,
                              win_loss]}

    output_key_df = pd.DataFrame(data=key_df_dict)
    # print(output_key_df)
    drawdown_df = MY_make_drawdown_df(df, number=5, fre=fre)
    # print(drawdown_df)
    return output_key_df, drawdown_df


def key_indicators_report(df, benchmark_df, benchmark_name, fre="周", rf=0.015):
    # full_df = pd.merge(df,benchmark_df,how='left')
    # full_df = pd.concat([df, benchmark_df], axis=1, join='inner')
    full_df1 = df.dropna()
    starttime = full_df1.index[0].strftime('%Y-%m-%d')
    endtime = full_df1.index[-1].strftime('%Y-%m-%d')
    key_report, drawdown_report = get_indicators(df, benchmark_df, benchmark_name, fre=fre, rf=rf)
    return key_report, drawdown_report, starttime, endtime


def get_excess_indicators(df, benchmark_df, fre="周", rf=0.015):
    df1 = pd.concat([df, benchmark_df], axis=1, join='inner')
    # df_excess1_nev: ret (Difference approach)
    df_excess1_nev = MY_get_excess_nev1(df1)
    # df_excess2_nev: vol, maxdd (Product approach)
    df_excess2_nev = MY_get_excess_nev2(df1)
    # trading_period = len(df.index.tolist())
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
        maxdd_six_mons = format(-MY_maxDD_back_period(df_excess2_nev, 180)[0].round(4), '.2%')
    except:
        maxdd_six_mons = format(
            -MY_maxDD_back_period(df_excess2_nev, (df_excess2_nev.index[-1] - df_excess2_nev.index[0]).days)[0].round(
                4), '.2%')

    try:
        maxdd_2021 = format(
            -MY_maxDD_back_period(df_excess2_nev, (df_excess2_nev.index[-1] - datetime.datetime(2021, 1, 1)).days)[0].round(4),
            '.2%')
    except:
        maxdd_2021 = format(
            -MY_maxDD_back_period(df_excess2_nev, (df_excess2_nev.index[-1] - df_excess2_nev.index[0]).days)[0].round(
                4), '.2%')

    maxdd_interval = MY_max_drawdown_trim(df_excess2_nev)
    maxdd_start = str(maxdd_interval[0])
    maxdd_end = str(maxdd_interval[1])
    if fre == '周':
        maxdd_lasting_periods = (
                datetime.datetime.strptime(maxdd_end, '%Y-%m-%d') - datetime.datetime.strptime(maxdd_start, '%Y-%m-%d')).days
        maxdd_lasting_periods = round(maxdd_lasting_periods / 7)
        maxdd_lasting_periods = str(maxdd_lasting_periods) + fre
    else:
        maxdd_lasting_periods = (
                datetime.datetime.strptime(maxdd_end, '%Y-%m-%d') - datetime.datetime.strptime(maxdd_start, '%Y-%m-%d')).days
        maxdd_lasting_periods = str(maxdd_lasting_periods) + fre
    ann_vol = format(MY_ann_vol(df_excess2_nev, fre=fre).round(4), '.2%')
    ann_vol_down = format(MY_ann_vol_down(df_excess2_nev, fre=fre).round(4), '.2%')
    df_excess1_nev_1yr = df.loc[(df.index[-1] - timedelta(days=365)):df.index[-1], :]
    df_excess2_nev_1yr = df.loc[(df.index[-1] - timedelta(days=365)):df.index[-1], :]
    try:
        sharpe_back_1yr = MY_Sharpe_excess(df_excess1_nev_1yr, df_excess2_nev_1yr, rf=rf, fre=fre)[0].round(2)
    except:
        sharpe_back_1yr = '/'
    sharpe = MY_Sharpe_excess(df_excess1_nev, df_excess2_nev, rf=rf, fre=fre)[0].round(2)
    calmar = MY_Calmar_excess(df_excess1_nev, df_excess2_nev)[0].round(2)
    sortino = MY_Sortino_excess(df_excess1_nev, df_excess2_nev, rf=rf, fre=fre)[0].round(2)
    win_loss = MY_win_loss(df_excess1_nev)[0].round(2)
    win_rate = format(MY_win_rate(df_excess1_nev)[0].round(4), '.2%')
    # print(df.columns[0])

    excess_df_dict = {'name1': ['区间累计收益率', '近一个月收益率', '近六个月收益率', '最大回撤', '最大回撤波峰',
                                "单" + str(fre) + "最大回撤", '近六月最大回撤', '年化波动率', '近一年夏普比率',
                                '卡玛比率', '盈亏比率'],
                      'value1': [period_acc_ret, rec_one_mon, rec_six_mons, maxdd, maxdd_start, period_maxdd,
                                 maxdd_six_mons, ann_vol, sharpe_back_1yr, calmar, win_loss],
                      'name2': ['年化收益率', '近三个月收益率', '近一年收益率', '回撤持续时间', '最大回撤波谷',
                                '近三月最大回撤', '2021年最大回撤', '年化下行波动率', '统计区间夏普比率',
                                '索提诺比率', str(fre) + '胜率'],
                      'value2': [ann_ret, rec_three_mons, rec_one_year, maxdd_lasting_periods, maxdd_end,
                                 maxdd_three_mons,
                                 maxdd_2021, ann_vol_down, sharpe, sortino, win_rate]
                      }
    output_excess_df = pd.DataFrame(data=excess_df_dict)
    excess_drawdown_df = MY_make_drawdown_df(df_excess2_nev, number=5, fre=fre)

    return output_excess_df, excess_drawdown_df, df_excess1_nev, df_excess2_nev


def excess_indicators_report(df, benchmark_df, fre="周", rf=0.015):
    full_df1 = df.dropna()
    excess_report, excess_drawdown_df, df_excess1_nev, df_excess2_nev = get_excess_indicators(full_df1, benchmark_df,
                                                                                              fre=fre, rf=rf)
    return excess_report, excess_drawdown_df, df_excess1_nev, df_excess2_nev


def get_monthly_ret_df(df, plot=False, float_data=False):
    df1 = df.copy()
    df2 = df.copy()
    name = df.columns[0]
    first_mon_start_value = df.iloc[0].values[0]
    df1 = df_standard_single(df1, fre='月')
    #print('#######################')
    #print(df1)
    first_mon_end_value = df1.iloc[0].values[0]
    df_ret = df1.pct_change().fillna(first_mon_end_value / first_mon_start_value - 1)
    df_ret['date'] = df_ret.index
    df_ret['year'] = df_ret['date'].dt.year
    df_ret['month'] = df_ret['date'].dt.month
    year_list = df_ret['year'].drop_duplicates().tolist()
    mon_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    month_index_df = pd.DataFrame(index=mon_list)
    mon_ret_container = []
    mon_ret_plot_df_list = []
    for i in year_list:
        sub_df_ret = df_ret[df_ret['year'] == i]
        sub_df_ret = sub_df_ret[[name, 'month']]
        sub_df_ret.set_index('month', inplace=True)
        sub_df_ret = month_index_df.join(sub_df_ret, how='left')
        sub_df_ret2 = sub_df_ret
        sub_df_ret2['mon_ret'] = sub_df_ret2[name]
        sub_df_ret2['年'] = i
        sub_df_ret2 = sub_df_ret2[['mon_ret', '年']]
        sub_df_ret2['month'] = np.array([str(i) + '月' for i in mon_list])
        # print(sub_df_ret2)
        mon_ret_plot_df_list.append(sub_df_ret2)
        if float_data == True:
            acc_this_yr_ret = MY_year_return(df2, i)[0]
            sub_df_ret = sub_df_ret.round(4)
            ret_values = sub_df_ret[name].tolist()
        else:
            if type(MY_year_return(df2, i)[0]) != str:
                acc_this_yr_ret = format(MY_year_return(df2, i)[0], '.2%')
            else:
                acc_this_yr_ret = MY_year_return(df2, i)[0]
            sub_df_ret = sub_df_ret.fillna('/').round(4)
            ret_values = sub_df_ret[name].apply(lambda x: ('%.2f%%' % (x * 100)) if type(x) == float else '/').tolist()
        ret_values.append(acc_this_yr_ret)
        mon_ret_container.append(ret_values)
    mon_list = [str(i) + '月' for i in mon_list]
    mon_list.append('累计')
    # print(mon_list)
    mon_ret_plot_df = pd.concat(mon_ret_plot_df_list, axis=0, ignore_index=True)
    # print(mon_ret_plot_df)
    if plot == True:
        plt.figure(figsize=(12, 4))
        sns.barplot(x='month', y='mon_ret', data=mon_ret_plot_df, hue='年')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.legend(loc='upper left')
        plt.title('月度收益')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.tight_layout()
        plt.show()
    else:
        pass
    monthly_ret_df = pd.DataFrame(index=year_list, columns=mon_list, data=np.array(mon_ret_container))
    return monthly_ret_df


def get_monthly_excess_ret_df(df, benchmark_df):
    df1 = df.copy()
    benchmark_mon_ret = pd.DataFrame(index=df1.index).join(benchmark_df, how='left')
    df_mon_ret = get_monthly_ret_df(df1, plot=False, float_data=True)
    benchmark_mon_ret = get_monthly_ret_df(benchmark_mon_ret, plot=False, float_data=True)
    excess_mon_ret = df_mon_ret - benchmark_mon_ret
    excess_mon_ret_plot_df = excess_mon_ret.iloc[:, :-1].T
    excess_mon_ret = excess_mon_ret.fillna('/')
    excess_mon_ret = excess_mon_ret.applymap(lambda x: '%.2f%%' % (x * 100) if type(x) == float else '/')
    excess_mon_ret_plot_df_list = []
    for i in excess_mon_ret_plot_df.columns:
        sub_excess_mon_ret_plot_df = excess_mon_ret_plot_df[[i]]
        sub_excess_mon_ret_plot_df['month'] = sub_excess_mon_ret_plot_df.index
        sub_excess_mon_ret_plot_df['年'] = i
        sub_excess_mon_ret_plot_df['mon_ret'] = sub_excess_mon_ret_plot_df[i]
        sub_excess_mon_ret_plot_df = sub_excess_mon_ret_plot_df[['mon_ret', 'month', '年']]
        excess_mon_ret_plot_df_list.append(sub_excess_mon_ret_plot_df)
    mon_excess_ret_plot_df = pd.concat(excess_mon_ret_plot_df_list, axis=0, ignore_index=True)
    # print(mon_excess_ret_plot_df)
    plt.figure(figsize=(12, 4))
    sns.barplot(x='month', y='mon_ret', data=mon_excess_ret_plot_df, hue='年')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend(loc='upper left')
    plt.title('月度超额收益')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tight_layout()
    plt.show()
    return excess_mon_ret


def get_rolling_df(df, rolling_windows, fre, rf=0.015):
    df1 = df.copy()
    df1_ret = df1.pct_change()
    name = df1.columns[0]
    if fre == '周':
        if rolling_windows > 36:
            ann_fre_windows = 52
            ann_fre_windows_name = '一年'
        elif rolling_windows > 18:
            ann_fre_windows = 26
            ann_fre_windows_name = '六个月'
        elif rolling_windows > 9:
            ann_fre_windows = 13
            ann_fre_windows_name = '三个月'
        else:
            ann_fre_windows = 4
            ann_fre_windows_name = '一个月'
    else:
        if rolling_windows > 36:
            ann_fre_windows = 250
            ann_fre_windows_name = '一年'
        elif rolling_windows > 18:
            ann_fre_windows = 125
            ann_fre_windows_name = '六个月'
        elif rolling_windows > 9:
            ann_fre_windows = 63
            ann_fre_windows_name = '三个月'
        else:
            ann_fre_windows = 21
            ann_fre_windows_name = '一个月'
    # rolling_df = df1.pct_change(ann_fre_windows).dropna()
    if fre == '周':
        rolling_df = np.power((df1.pct_change(ann_fre_windows) + 1).dropna(), 52 / ann_fre_windows) - 1
    else:
        rolling_df = np.power((df1.pct_change(ann_fre_windows) + 1).dropna(), 250 / ann_fre_windows) - 1
    rolling_df['过去' + str(ann_fre_windows_name) + '滚动收益率'] = rolling_df[name]
    rolling_df['过去' + str(ann_fre_windows_name) + '滚动波动率'] = df1_ret.rolling(
        window=ann_fre_windows).std().dropna() * np.sqrt(ann_fre_windows)
    rolling_df['过去' + str(ann_fre_windows_name) + '滚动夏普比率(右轴)'] = (rolling_df[name] - rf) / rolling_df[
        '过去' + str(ann_fre_windows_name) + '滚动波动率']
    rolling_df = rolling_df[['过去' + str(ann_fre_windows_name) + '滚动收益率', '过去' + str(ann_fre_windows_name) + '滚动波动率',
                             '过去' + str(ann_fre_windows_name) + '滚动夏普比率(右轴)']]
    return rolling_df


def get_dynamic_drawdown(df,excess=False):
    df1 = df.copy()
    name = df.columns[0]
    df1['max_back_values'] = np.array(MY_find_max_back_value_list(df1[name].tolist()))
    if excess==True:
        df1['超额动态回撤'] = df1[name] / df1['max_back_values'] - 1
        df1 = df1[['超额动态回撤']]
    else:
        df1['动态回撤'] = df1[name] / df1['max_back_values'] - 1
        df1 = df1[['动态回撤']]
    return df1


def get_dynamic_drawdown_df(df, benchmark_df, df_excess2_nev):
    df_self = df.copy()
    df1 = df.copy()
    full_df = df1.join(benchmark_df, how='left')
    name_list = full_df.columns.tolist()
    full_df1 = full_df.dropna()
    full_df1 = full_df1.apply(lambda x: MY_acc_ret_df(x))
    full_df1['超额收益'] = full_df1[name_list[0]] - full_df1[name_list[1]]
    full_df1 = full_df1.iloc[1:, :]
    dynamic_excess_drawdown_df = get_dynamic_drawdown(df_excess2_nev,excess=True)
    dynamic_self_drawdown_df = get_dynamic_drawdown(df_self, excess=False)
    dynamic_drawdown_df_full = full_df1.join(dynamic_self_drawdown_df, how='left')
    dynamic_drawdown_df_full = dynamic_drawdown_df_full.join(dynamic_excess_drawdown_df, how='left')
    return dynamic_drawdown_df_full


def MY_plot(df, benchmark_data, sheetname, dates_day, dates_week, dates_month, workbook, fre="周",
                           rf=0.015, excess=True, write=False):
    benchmark_df = benchmark_data
    benchmark_df = df_standard_single(benchmark_df, dates_day, dates_week, dates_month, fre=fre)
    




def test():
    w.start()
    w.isconnected()
    dates_day = get_tradedates(type='日')
    dates_week = get_tradedates(type='周')
    dates_month = get_tradedates(type='月')
    testdata = pd.read_excel('test.xlsx', index_col=0)
    testdata.index = pd.to_datetime(testdata.index)
    testdata = testdata.dropna()
    fre = '周'
    benchmark_code = '000300.SH'
    benchmark_name = '沪深300'
    ogdata = testdata
    print(ogdata)
    benchmark_data = get_benchmark(benchmark_code, benchmark_name)
    
