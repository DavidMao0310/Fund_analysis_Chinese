import pandas as pd
import time
import warnings
import xlwings as xw
from WindPy import w
w.start()
w.isconnected()
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

#get_wind_data('001938.OF')

def fill_data():
    wb = xw.Book.caller()
    id =  wb.sheets[0].range('P6').value
    df = get_wind_data(id)
    fund_name = df.columns[0]
    df = df.reset_index(drop=False)
    wb.sheets[0].range('M2').options(expand='table').value = df.values
    wb.sheets[0].range('N1').options(expand='table').value = fund_name
    wb.sheets[0].range('M1').options(expand='table').value = 'date'




