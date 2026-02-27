import requests
def get_btc_price():
    url = "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
    r = requests.get(url)
    if r.status_code == 200:
        return float(r.json()["price"])
    return None

if __name__ == "__main__":
    print(get_btc_price())
