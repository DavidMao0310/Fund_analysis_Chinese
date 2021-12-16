import datetime
import numpy as np
import pandas as pd
from datetime import timedelta
import statsmodels.api as sm
from statsmodels import regression


def judge_number_sign(x, set='positive'):
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


def count_accelerated_number(given_series):
    container = []
    j = 0
    for i in given_series:
        if i == 1:
            j += 1
        else:
            j = 0
        container.append(j)
    return np.array(container)


def count_accelerated_period_return(given_series):
    container = []
    j = 1
    for i in given_series:
        if i != 0:
            j = (1 + i) * j
        else:
            j = 1
        container.append(j)
    return container


def get_step_after_fee(df, fee):
    df = df / df.iloc[0]
    df_after_fee = pd.DataFrame()
    # 获取扣费日期
    years = int((df.index[-1] - df.index[0]).days / 365)
    date_fee = []
    for i in range(years + 1):
        date = df.index[0] + datetime.timedelta(days=365 * (i))
        date = df[date:].index[0]
        date_fee.append(date)

    if date == df.index[-1]:
        pass
    else:
        date_fee.append(df.index[-1])

    # 根据分红日期进行分组，计算费后虚拟净值
    for i, date1 in enumerate(date_fee[:-1]):
        date2 = date_fee[i + 1]
        df_sub = df[date1:date2]
        df_after = after_fee(df_sub, fee)
        df_after_ret = (df_after / df_after.shift(1) - 1).dropna()
        df_after_fee = pd.concat([df_after_fee, df_after_ret])
    df_after_fee = (df_after_fee + 1).cumprod()
    df_join = df.join(df_after_fee, how='left').fillna(1)
    # 若需要仅保留费前净值，取消注释即可
    # df_join = df.iloc[:,[1]]
    df_join = df_join[df_join.columns[1]]
    return df_join


def after_fee(df, fee):
    """
    输入：
        df: dataframe,原始净值数据，数据短于1年
        fee: list,管理费比率，依次为为开始扣费的基础和管理报酬计提比例
             可设置2,4,6,8各参数，表示不同的收费梯度(现在市场上收费标准最多为4个梯度)
    输出：
        df:费后净值
    """
    length = len(fee)

    if length == 2:
        start1, fee1 = fee
        start4, fee4 = start3, fee3 = start2, fee2 = start1, fee1
    elif length == 4:
        start1, start2, fee1, fee2 = fee
        start4, fee4 = start3, fee3 = start2, fee2
    elif length == 6:
        start1, start2, start3, fee1, fee2, fee3 = fee
        start4, fee4 = start3, fee3
    else:
        start1, start2, start3, start4, fee1, fee2, fee3, fee4 = fee

    # 运行天数，计算年化收益
    df = df / df.iloc[0]
    acc_days = np.array((df.index - df.index[0]).days)
    df_ = df.iloc[:, 0].values

    num = len(df_)
    for i in range(num):
        # 根据不同的日期计算扣费基准的年化收益
        ann_coef = acc_days[i]
        s1 = (start1 + 1) ** (ann_coef / 365)
        s2 = (start2 + 1) ** (ann_coef / 365)
        s3 = (start3 + 1) ** (ann_coef / 365)
        s4 = (start4 + 1) ** (ann_coef / 365)

        if s1 < df_[i] <= s2:
            df_[i] = df_[i] - (df_[i] - 1) * fee1
        elif s2 < df_[i] <= s3:
            df_[i] = df_[i] - (df_[i] - s2) * fee2 - (s2 - s1) * fee1
        elif s3 < df_[i] <= s4:
            df_[i] = df_[i] - (df_[i] - s3) * fee3 - \
                     (s2 - s1) * fee1 - (s3 - s2) * fee2
        elif df_[i] > s4:
            df_[i] = df_[i] - (df_[i] - s4) * fee4 - (s2 - s1) * \
                     fee1 - (s3 - s2) * fee2 - (s4 - s3) * fee3
        else:
            pass
    df_ = pd.DataFrame(df_, index=df.index, columns="费后-" + df.columns)

    return df_


def modify_high_freq(x, const, rate):
    if x / const - 1 > 0:
        y = x - rate * (x / const - 1)
    else:
        y = x
    return y


def get_single_after_fee(df, rate=0.5):
    df1 = df.copy()
    df1.dropna(inplace=True)
    df1.index = pd.to_datetime(df1.index)
    df1['归一'] = df1[df1.columns[0]] / df1[df1.columns[0]][0]
    df1['费后-' + df1.columns[0]] = df1['归一'].apply(lambda x: modify_high_freq(x, df1['归一'][0], rate))
    # df1.drop(columns='归一', inplace=True)
    df1 = df1['费后-' + df1.columns[0]]
    return df1


def get_excess_df(df):
    '''
    with benchmark in the last column
    '''
    df_col = df.columns.tolist()
    for i in df_col:
        df[i] = df[i] - df[df_col[-1]]
    df = df.iloc[:, :-1]
    return df


def get_excess_nev1(df):
    '''
    with benchmark in the last column
    '''
    df = df / df.iloc[0]
    df1 = get_excess_df(df) + 1
    return df1


def get_excess_nev2(df):
    '''
    with benchmark in the last column
    '''
    df = df.pct_change()
    df1 = get_excess_df(df) + 1
    df1 = df1.cumprod()
    df1.fillna(1, inplace=True)
    return df1


def ret(df):
    """日/周收益率"""
    return (df / df.shift(1)).dropna() - 1


def acc_ret(df):
    """累计收益率"""
    return df.values[-1] / df.values[0] - 1


def acc_ret_df(df):
    """累计收益率序列"""
    return df / df.iloc[0] - 1


def ann_ret(df):
    """年化收益率"""
    period = df.index[-1] - df.index[0]
    return pow(acc_ret(df) + 1, 365 / period.days) - 1


def ret_to_nev(df_ret):
    """根据收益算净值"""
    return (df_ret + 1).cumprod()


def rec_week_ret(df, num):
    """
    计算近几周收益率
    num为周数
    """
    date = df.index[-1]
    df_week = df.tail(num + 1)
    return acc_ret(df_week)


def rec_interval_ret(df, period):
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


def year_return(df, year):
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
        year_return = acc_ret(df)
    except:
        year_return = ['/']
    return year_return


def ann_para(fre="周"):
    """年化参数"""
    if fre == "日":
        ann_para = 250
    elif fre == "周":
        ann_para = 52
    else:
        ann_para = 12
    return ann_para


def ann_vol(df, fre="周"):
    """年化波动率"""
    ann_coff = ann_para(fre)
    return ret(df).values.std() * np.sqrt(ann_coff)


def ann_vol_down(df, fre="周"):
    """年化下行波动率"""
    ret_ = ret(df)
    ret_down = (ret_[ret_ < 0]).dropna().values
    ann_coff = ann_para(fre)
    return ret_down.std() * np.sqrt(ann_coff)


def maxDD(df):
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


def maxDD_sigle(df):
    """
    计算单期最大回撤
    """
    return -ret(df).min().values


def maxDD_back_period(df, backdays):
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
    maxdd1 = maxDD(backdate)

    return maxdd1


def max_drawdown_trim(df):
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


def find_drawdown_interval(df):
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


def make_drawdown_df(df, number, fre='周'):
    df_copy = df.copy()
    container = []
    for i in range(number):
        maxdd_start, maxdd_end, rate, df_copy = find_drawdown_interval(df_copy)
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


def Sharpe(df, rf=0.015, fre="周"):
    """夏普比率"""
    return (ann_ret(df) - rf) / ann_vol(df, fre)


def Sharpe_excess(df1, df2, rf=0.015, fre="周"):
    '''
    :param df1: ret 超额净值序列
    :param df2: vol, maxdd 超额净值序列
    :return: a single number
    '''
    return (ann_ret(df1) - rf) / ann_vol(df2, fre)


def Sortino(df, rf=0.015, fre="周"):
    """索提诺比率"""
    return (ann_ret(df) - rf) / ann_vol_down(df, fre)


def Sortino_excess(df1, df2, rf=0.015, fre="周"):
    '''
    :param df1: ret 超额净值序列
    :param df2: vol, maxdd 超额净值序列
    :return: a single number
    '''
    return (ann_ret(df1) - rf) / ann_vol_down(df2, fre)


def Calmar(df):
    """卡玛比率"""
    return ann_ret(df) / maxDD(df)


def Calmar_excess(df1, df2):
    '''
    :param df1: ret 超额净值序列
    :param df2: vol, maxdd 超额净值序列
    :return: a single number
    '''
    return ann_ret(df1) / maxDD(df2)


def win_loss(df):
    """
    计算盈亏比率
    """
    retdf = ret(df)
    df = df.shift(1).dropna()
    df_ret = df * retdf
    return -(df_ret[df_ret > 0].sum() / df_ret[df_ret < 0].sum()).values


def win_rate(df):
    """
    计算胜率
    """
    df_ret = ret(df)
    return (df_ret[df_ret >= 0].count() / df_ret.count()).values


def linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    # r_sq = model.rsquared
    return model.params[0], model.params[1]


def self_linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    r_sq = model.rsquared
    return model.params[0], model.params[1], r_sq


def get_none_matrix(row_num, col_num):
    a = []
    b = []
    for i in range(col_num):
        a.append(None)
    for i in range(row_num):
        b.append(a)
    return b


def find_max_back_value_list(l):
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
