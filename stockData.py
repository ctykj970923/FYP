from setup import *


def getStockData(stock):
    STOCK = stock
    date_now = dt.datetime.now()

    date_3_years_back = (dt.date.today() - dt.timedelta(days=3104)).strftime("%Y-%m-%d")

    initialDataFrame = yff.get_data(
        STOCK, start_date=date_3_years_back, end_date=date_now, interval="1d"
    )

    initialDataFrame = initialDataFrame.drop(
        ["open", "high", "low", "adjclose", "ticker", "volume"], axis=1
    )

    initialDataFrame["date"] = initialDataFrame.index

    return initialDataFrame
