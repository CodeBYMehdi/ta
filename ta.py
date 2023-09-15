

# Import the IB modules

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.common import *
from ibapi.ticktype import *
from ibapi.order import *
from ib_insync import *



# Import classic modules


import requests
import exchange
import json
import time
from datetime import datetime, timedelta
import logging
import uuid
import numpy as np
import ta
import matplotlib.pyplot as plt
import pandas as pd
import threading
import pandas_datareader.data as web
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import the machine learning

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import Deep Learning modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam

# Import the dataset module

from ib_insync import IB, Forex, Stock, util

class Market:
    def __init__(self):
        self.ib = IB()
        self.connect_to_tws()
    def connect_to_tws(self):
        try:

            self.ib.connect("127.0.0.1", 7495, clientId=1)  
           

            if self.ib.isConnected():
                print("Connected to TWS")
            else:
                print("Connection to TWS failed")

        except Exception as e:
            print(f"Error connecting to TWS: {e}")

    def get_historical_data(self, symbol, duration, bar_size, end_datetime=None):
        if '.' in symbol:
            contract = Forex(symbol)
        else:
            contract = Stock(symbol, 'MKT', 'EUR')

        request = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        util.waitUntil(lambda: len(request) > 0)

        df = util.df(request)

        return df

    def get_real_time_data(self, symbol):
        if '.' in symbol:
            contract = Forex(symbol)
        else:
            contract = Stock(symbol, 'SMART', 'USD')

        self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(5)

        ticker = self.ib.ticker(contract)
        self.ib.cancelMktData(contract)

        return ticker





        


class IBapi(EWrapper, EClient):
    def __init__(self, bot):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.bot = bot


class BalanceApp(EWrapper, EClient, float):
    
    def __init__(self, ip_address, port_id, client_id):
        EClient.__init__(self, self)
        self.ip_address = ip_address
        self.port_id = port_id
        self.client_id = client_id
        self.account_balance = None
    
    def __new__(cls, ip_address, port_id, client_id):
        return float.__new__(cls, 0.0)
    

    def start(self):
        self.connect(self.ip_address, self.port_id, self.client_id)
        self.run()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        super().accountSummary(reqId, account, tag, value, currency)
        if tag == 'TotalCashValue':
            self.account_balance = float(value)

    def __float__(self):
        if self.account_balance:
            return float(self.account_balance)
        else:
            return 0.0

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} - {errorCode} - {errorString}")
        if errorCode == 2104:  
            return

        





class RiskManager:
    def __init__(self, balance, max_loss_pct, stop_loss_pct, take_profit_pct):
        self.balance = balance  # This is the BalanceApp instance
        self.max_loss_pct = max_loss_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_loss_pct = 0.04  # maximum percentage of account balance that can be lost on a single trade
        self.take_profit_pct = self.calculate_max_take_profit_pct()
        
    def calculate_max_take_profit_pct(self):
        # Fetch the balance from the BalanceApp instance and convert it to a float
        actual_balance = float(self.balance)
        
        return self.max_loss_pct / (1 - self.stop_loss_pct)

    def calculate_order_size(self, current_price):
        
        if not isinstance(self.balance, (int, float)):
            raise ValueError("Balance must be a numeric value")
        
        if not isinstance(self.max_loss_pct, float):
            raise TypeError("Max loss percentage must be a float")
        
        risk_amount = float(self.balance) * self.max_loss_pct
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        take_profit_price = current_price * (1 + self.take_profit_pct)
        
        order_size = risk_amount / (current_price - stop_loss_price)
        
        potential_profit = order_size * (take_profit_price - current_price)
        
        if potential_profit < risk_amount:
            order_size = risk_amount / (take_profit_price - current_price)
            
        return int(order_size)

        
    def calculate_risk(self, price, stop_loss):
        risk = self.balance * self.max_loss_pct
        max_loss = price - stop_loss
        position_size = risk / max_loss

        return position_size
    






class NNTS:
    
    def __init__(self, lookback, units, dropout, epochs, batch_size):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.risk_manager = RiskManager(balance=BalanceApp(ip_address="127.0.0.1", port_id=7495, client_id=1), stop_loss_pct=0.03, max_loss_pct=0.04, take_profit_pct=0.05)

    def _prepare_data(self, data):
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def _build_model(self, X):
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        return model

    def generate_signals(self, data, strategy, max_trades=300):
        avg_trades_per_day = int(len(data) / self.lookback)
        if avg_trades_per_day < 250:
            self.units *= 2
        elif avg_trades_per_day > 300:
            self.lookback = int(len(data) / 300)
        batch_size = max(int(avg_trades_per_day / self.epochs), 1)
        X, y = self._prepare_data(data)
        model = self._build_model(X)
        model.fit(X, y, epochs=self.epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X)
        signals = np.zeros(len(data))
        signals[self.lookback:] = np.where(y_pred > y, 1, -1)
        signals = self.risk_manager.filter_signals(signals, data)

        if strategy == 'buy':
            signals[signals != 1] = 0
        elif strategy == 'sell':
            signals[signals != -1] = 0

        if np.count_nonzero(signals) > max_trades:
            excess_trades = np.count_nonzero(signals) - max_trades
            if excess_trades < np.count_nonzero(signals == 1):
                signals[signals == 1][:excess_trades] = 0
            else:
                signals[signals == -1][:excess_trades] = 0

        return signals





class TradingProcess:
    def __init__(self, balance, risk_percentage):
        self.balance = balance
        self.risk_percentage = risk_percentage
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)

        self.positions = []
        self.profits = []

    def update_equity(self):
        equity = self.balance
        for position in self.positions:
            equity += position['profit']
        return equity


    def can_open_position(self, price, stop_loss):
        position_size = self.calculate_risk(price, stop_loss)
        return self.balance >= position_size * price

    def can_afford_position(self, price, size):
        position_cost = size * price
        return self.balance >= position_cost

    def open_position(self, price, stop_loss, size):
        self.balance -= size * price
        self.positions.append({
            'price': price,
            'stop_loss': stop_loss,
            'size': size,
            'profit': 0.0
        })

    def close_position(self, index, price, current_signal):
        position = self.positions[index]
    
    # Check if the current signal is opposite to the signal at the time of opening the position
        if (current_signal == 'buy' and position['signal'] == 'sell') or \
        (current_signal == 'sell' and position['signal'] == 'buy'):
            profit = position['size'] * (price - position['price'])
            self.balance += profit
            self.profits.append(profit)
            self.positions.pop(index)
            return profit
        else:
        # Position remains open if signals are not opposite
            position['profit'] = position['size'] * (price - position['price'])
            return position['profit']


    def update_position(self, index, price):
        position = self.positions[index]
        if price <= position['stop_loss']:
            return self.close_position(index, position['stop_loss'])
        else:
            position['profit'] = position['size'] * (price - position['price'])
            return position['profit']

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)






class DataProcessor:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        # Drop any rows with NaN values
        data.dropna(inplace=True)

        # Add technical indicators
        data = ta.add_all_ta_features(data, "open", "high", "low", "close", "volume")

        # Create X and y
        X = data[self.feature_columns].values
        y = np.where(data["close"].shift(-1) > data["close"], 1, -1)
        y = y[:-1]

        # Scale the data
        X = self.scaler.fit_transform(X)

        return X, y





        




class PlaceCancelOrder:
    
    def __init__(self):
        self.orders = []
        self.units = None


    def place_order(self, buy_signals, sell_signals, symbol, order_type):
        self.units = self.calculate_units()
        if buy_signals == 'buy':
            self.orders.append({'side': 'buy',
                                'units': self.units,
                                'strategy': 'NNTS',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif sell_signals == 'sell':
            self.orders.append({'side': 'sell',
                                'units': self.units,
                                'strategy': 'NNTS',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})


    def cancel_order(self, order_id):
        request = orders.OrderCancel(self.account_id, orderID=order_id)
        self.client.request(request)
        for order in self.orders:
            if order['id'] == order_id:
                self.orders.remove(order)
                break
    






class Bot:
    ib = None
    
    def __init__(self):
        self.ib = IBapi(self)
        self.ib.connect("127.0.0.1", 7495, 1)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)

    def connectAck(self):
        print("Connected to TWS")

    def execute_trade(self, buy_signals, sell_signals, quantity, price):
        # create a new contract object
        print("execute")
        print("execute trade")
        contract = Contract()
        contract.symbol = 'EURUSD'
        contract.secType = "forex"  
        contract.currency = "EUR"  
        contract.exchange = "MKT"  

        order = Order() 
        if buy_signals == 'BUY':
            order.action = 'BUY'
        if sell_signals == 'SELL':
            order.action = 'SELL'
        order.orderType = 'MKT' 
        order.totalQuantity = quantity
        order.lmtPrice = price 
    
        # submit the order to the TWS
        self.ib.placeOrder(self.ib.nextOrderId, contract, order)
        self.ib.nextOrderId += 1
    
        # wait for the order to be filled
        time.sleep(1)
        print("sleep")
    
        # cancel the unfilled portion of the order
        remaining_quantity = order.totalQuantity - order.filledQuantity
        if remaining_quantity > 0:
            cancel_order = Order()
            cancel_order.action = "CANCEL"
            cancel_order.totalQuantity = remaining_quantity
            self.ib.placeOrder(self.ib.nextOrderId, contract, cancel_order)
            self.ib.nextOrderId += 1

        today = datetime.datetime.today()
        if today.weekday() < 5: 
            # Execute the trade
            signal = generate_signals()
            if signal == 'BUY':
                print("Mechi position long...")
            elif signal == 'SELL':
                print("Mechi position short...")
            else:
                print("Ma fomech signal...")
        else:
            print("Lioum Weekend ma fomech des signaux...")
        
    def run_loop(self):
        self.ib.run()

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from TWS")





#run loop

print("test 1")

if __name__ == "__main__":
    try:
        market = Market()
        
        msft_historical_data = market.get_historical_data('MSFT', '1 Y', '1 day')
        msft_real_time_data = market.get_real_time_data('MSFT')
        eurusd_historical_data = market.get_historical_data('EUR.USD', '1 Y', '1 day')
        eurusd_real_time_data = market.get_real_time_data('EUR.USD')
    
    except ConnectionError as e:
        print(f"Erreur de connexion : {e}")



ip_address = "127.0.0.1" 
port_id = 7495 
client_id = 1  
current_price=market.fx_price(real_time= True)
print(current_price);
price=eurusd_historical_data, eurusd_real_time_data
data=price
bot = Bot()


print("wa7el Houni");
balance = BalanceApp(ip_address,port_id,client_id)
balance.start()
balance.accountSummary(reqId=123, account="DU11643091", tag="TotalCashValue", value="12345", currency="EUR")
balance.__float__()
balance.error(reqId=123, errorCode=456, errorString="Some error message")

print('Is balance a float?', isinstance(balance, float))

print("test 5")

# Call the RiskManager class


riskmg = RiskManager(balance=BalanceApp(ip_address, port_id, client_id), max_loss_pct=0.04, stop_loss_pct=0.03, take_profit_pct=0.05)
max_take_profit_pct = riskmg.calculate_max_take_profit_pct()
print("Maximum take profit pct: ", max_take_profit_pct)
order_size=riskmg.calculate_order_size(current_price)
print("Order size:", order_size)
riskmg.calculate_risk(price, stop_loss=0.04)
print("test 6")

# Call the NNTS class

nnts = NNTS(lookback=50, units=128, dropout=0.5, epochs=200, batch_size=64)
X, y=nnts._prepare_data(data)
model=nnts._build_model(X)
buy_signals=nnts.generate_signals(data, strategy='buy')
sell_signals=nnts.generate_signals(data, strategy='sell')
print("test 7")

# Call the TradingProcess class

tp = TradingProcess(balance, risk_percentage=0.05)
tp.update_equity()
tp.can_open_position(price, stop_loss=0.04)
tp.can_afford_position(price)
tp.open_position(price, stop_loss=0.04)
tp.close_position(price)
tp.update_position(price)
tp.fit(X, y)
tp.predict(X)
print("test 8")

# Call the DataProcessor class

datapp = DataProcessor(feature_collumns=["open","high", "low", "close", "volume"])
datapp.preprocess_data(data)
print("test 9")

# Call PlaceCancelOrder class

pcorder = PlaceCancelOrder()
pcorder.place_order(buy_signals, sell_signals, symbol='EURUSD', order_type='MKT')
pcorder.cancel_order(order_id=1)
print("test 10")

# Call Bot function
bot.connectAck()
bot.execute_trade(buy_signals, sell_signals, price)



timestamps = np.arange(len(buy_signals, sell_signals, price))
plt.figure(figsize=(12,6))
plt.plot(timestamps, buy_signals, 'g^', label='Buy Signals', marketsize=8, markerfacecolor='none')
plt.plot(timestamps, sell_signals, 'rs', label='Sell Signals', marketsize=8, markerfacecolor='none')
plt.plot(timestamps, price, label='Price', marketsize=8, color='black', linewidth=2)
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Signals vs Price')
plt.grid(True)
plt.legend()

plt.show()

print("test 11")

bot.disconnect()
