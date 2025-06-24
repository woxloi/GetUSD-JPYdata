from flask import Flask, render_template, request
import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import base64
import datetime

app = Flask(__name__)

API_KEY = "832c33bc-7dea-4873-aaa6-99ca00991026"

# Windows環境の場合の例。LinuxやMacは適宜変更してください。
plt.rcParams["font.family"] = "Yu Gothic"

def get_usd_jpy_data():
    # Alpha Vantage 為替データ取得（FX_DAILY）
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=JPY&outputsize=compact&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "Time Series FX (Daily)" not in data:
        raise Exception("為替データが取得できませんでした。APIキーやAPI制限を確認してください。")

    ts = data["Time Series FX (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df['4. close'].astype(float)
    return df

def create_plot():
    data = get_usd_jpy_data()
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)

    plt.figure(figsize=(10,5))
    plt.plot(data.index, data, label='過去のUSD/JPY終値（実測値）')
    future_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, 11)]
    plt.plot(future_dates, forecast, label='将来10日間の予測値（ARIMAモデル）', color='red')
    plt.title("USD/JPY 為替レートの実測値とARIMAモデルによる予測")
    plt.xlabel("日付")
    plt.ylabel("価格 (JPY)")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return plot_img

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    plot_img = None

    try:
        plot_img = create_plot()
    except Exception as e:
        error = str(e)

    return render_template('index.html', plot_img=plot_img, error=error)

if __name__ == '__main__':
    app.run(debug=True)
