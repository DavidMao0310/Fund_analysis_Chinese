import pymysql
import numpy as np
import pandas as pd
import time
import warnings
import xlwings as xw
from WindPy import w

warnings.filterwarnings('ignore')


def get_wind_data(id):
    today = time.gmtime()
    today = time.strftime("%Y%m%d", today)
    windid = id
    date = w.wss(windid, "fund_setupdate,NAV_date", "tradeDate={}".format(today)).Data
    start_date = date[0][0].strftime('%Y-%m-%d')
    end_date = date[1][0].strftime('%Y-%m-%d')
    fund_name_list = w.wss(windid, "name_official,fund_fundmanager").Data
    fund_name = fund_name_list[0][0] +'('+ fund_name_list[1][0]+')'
    data = w.wsd(windid, "NAV_adj", start_date, end_date, "Period=W;Fill=Previous",usedf=True)[1]
    data = data.reset_index(drop=False)
    data['date'] = data['index']
    data[fund_name] = data['NAV_ADJ']
    data= data[['date',fund_name]]
    data = data.set_index('date')
    return data


def execude_sql(sql):
    # 创建连接
    try:
        db = pymysql.connect(host='106.75.45.237',
                             port=15630,
                             user='simu_tfzqzg',
                             passwd='nvj64PsxhfN5gHDf',
                             db='CUS_FUND_DB',
                             charset='utf8')
    except:
        print('数据库连接失败，3s后重试')
        time.sleep(3)
    # 创建游标
    cursor = db.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    fund_name = result[0][2]
    # 转成dataframe格式
    df = pd.DataFrame(result, columns=["日期", fund_name, '产品名称']).set_index('日期')
    df = df.loc[:, [fund_name]]
    df[fund_name] = df[fund_name].apply(lambda x: float(x))
    df.index = pd.to_datetime(df.index)
    # 关闭游标
    db.close()
    return df


def sql_data(id):
    sql = "SELECT statistic_date,swanav,fund_name FROM t_fund_nv_data_zyyx WHERE fund_id =" + str(id)
    df = execude_sql(sql)
    return df


# SQL：累计净值-added_nav   单位净值-nav   复权累计净值(按分红再投推算)-swanav

def get_all_data():
    wb = xw.Book.caller()
    og_df = wb.sheets['Get_data'].range('A1:C10').options(pd.DataFrame).value
    product_list = og_df[[og_df.columns[1]]].dropna()[og_df.columns[1]].tolist()
    start_col_list = ['E1', 'H1', 'K1', 'N1', 'Q1', 'T1', 'W1', 'Z1', 'AC1']
    j = 0
    for i in product_list:
        if type(i) == str:
            w.start()
            df = get_wind_data(i)
        else:
            df = sql_data(int(i))
        wb.sheets['Get_data'].range(start_col_list[j]).options(expand='table').value = df
        wb.sheets['Get_data'].range(start_col_list[j]).options(expand='table').value = 'date'
        j += 1
