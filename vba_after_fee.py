import pandas as pd
import numpy as np
from WindPy import w
import datetime
import xlwings as xw

w.start()
w.isconnected()


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
    df_join = df_join[[df_join.columns[1]]]
    return df_join


def df_standard_single(df, fre="周"):
    """
    数据频率标准化，fre可选日/周/月,默认取周
    """
    # df.index = pd.to_datetime(df.index)
    dates_day = get_tradedates(type='日')
    dates_week = get_tradedates(type='周')
    dates_month = get_tradedates(type='月')
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


def judge_bd_fee_value(x):
    if x == None:
        y = x
    else:
        y = float(x)
    return y


def get_single_after_fee():
    wb = xw.Book.caller()
    df = wb.sheets['收费'].range('D1:E6000').options(pd.DataFrame).value
    df = df.dropna()
    # df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    fre = wb.sheets['收费'].range('B3').value
    bd1 = 0
    bd2 = wb.sheets['收费'].range('A8').value
    bd3 = wb.sheets['收费'].range('A9').value
    bd4 = wb.sheets['收费'].range('A10').value
    fee1 = wb.sheets['收费'].range('C7').value
    fee2 = wb.sheets['收费'].range('C8').value
    fee3 = wb.sheets['收费'].range('C9').value
    fee4 = wb.sheets['收费'].range('C10').value
    value_container = [bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4]
    bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4 = [judge_bd_fee_value(i) for i in value_container]
    if bd2 == None:
        feelist = [bd1, fee1]
    elif bd3 == None:
        feelist = [bd1, bd2, fee1, fee2]
    elif bd4 == None:
        feelist = [bd1, bd2, bd3, fee1, fee2, fee3]
    else:
        feelist = [bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4]
    df1 = df_standard_single(df, fre=fre)
    df_afterfee = get_step_after_fee(df1, feelist)
    wb.sheets['收费'].range('G1').options(expand='table').value = df_afterfee


def tst():
    '''
    test algorithm
    :param df: 
    :return: 
    '''
    data = pd.read_excel(r'test.xlsx')
    df = data
    bd1 = 0
    bd2 = '0.08'
    bd3 = None
    bd4 = None
    fee1 = 0
    fee2 = '0.2'
    fee3 = 0.3
    fee4 = '0.6'
    value_container = [bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4]
    bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4 = [judge_bd_fee_value(i) for i in value_container]
    print([bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4])
    if bd2 == None:
        feelist = [bd1, fee1]
    elif bd3 == None:
        feelist = [bd1, bd2, fee1, fee2]
    elif bd4 == None:
        feelist = [bd1, bd2, bd3, fee1, fee2, fee3]
    else:
        feelist = [bd1, bd2, bd3, bd4, fee1, fee2, fee3, fee4]
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    df1 = df_standard_single(df, fre='周')
    df_afterfee = get_step_after_fee(df1, feelist)
    print(feelist)
    print(df1)
    print(df_afterfee)


