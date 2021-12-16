import pymysql
import numpy as np
import pandas as pd
import time
import warnings
import xlwings as xw

warnings.filterwarnings('ignore')


# 连接朝阳永续数据库提取产品净值数据
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

def fill_data():
    wb = xw.Book.caller()
    id = wb.sheets[0].range('O2').value
    df = sql_data(id)
    fund_name = df.columns[0]
    df = df.reset_index(drop=False)
    wb.sheets[0].range('M2').options(expand='table').value = df.values
    wb.sheets[0].range('N1').options(expand='table').value = fund_name
    wb.sheets[0].range('M1').options(expand='table').value = 'date'


def rangecopy():
    wb = xw.Book.caller()
    og_df = wb.sheets[0].range('M1:N6000').options(pd.DataFrame).value
    wb.sheets[0].range('B1:C6000').options(expand='table').value = og_df



def fill_data_datayes():
    wb = xw.Book.caller()
    id = wb.sheets[0].range('O3').value
    #df = get_data_datayes(id)
    df = pd.DataFrame()
    fund_name = df.columns[0]
    df = df.reset_index(drop=False)
    wb.sheets[0].range('M2').options(expand='table').value = df.values
    wb.sheets[0].range('N1').options(expand='table').value = fund_name
    wb.sheets[0].range('M1').options(expand='table').value = 'date'
