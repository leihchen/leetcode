
import datetime
from typing import *


# Trade1: (date = '2022-03-15', time=9:01:00, type=Broker, qty=-500, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE,
#         trade-id = 737acm, product=ABC)
# Trade2: (date = '2022-03-15', time=9:00:24, type=Electronic, qty=-200, strike=1500, expiry='2022-04-28', kind=P
#         exchange=CBOE, trade-id = w6c229, product=ABC)
# Trade3: (date = '2022-03-15', time=9:03:45, type=Electronic, qty=-100, strike=1500, expiry='2022-04-28', kind=P,
#         exchange=CBOE, trade-id = tssrin, product=ABC) [Falls condition (b)]
# Trade4: (date = '2022-03-15', time=9:00:53, type=Electronic, qty=-500, strike=1500, expiry='2022-04-28', kind=P,
#         exchange=CBOE, trade-id = lk451a, product=XYZ) [Fails condition (c)]
# Trade5: (date = '2022-03-15', time=9:00:05, type=Electronic, qty=-350, strike=1500, expiry='2022-04-28', kind=C,
#         exchange=CBOE, trade-id = 9numpr, product=ABC) [Fails condition (d)]
# Trade6: (date = '2022-03-15', time=9:00:35, type=Electronic, qty=200, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE,
#         trade-id = 922v3g, product=ABC) [Fails condition (e)]
# Trade7: (date = '2022-03-15', time=9:00:47, type=Electronic, qty=-150, strike=1500, expiry='2022-04-21', kind=P
#         exchange=CBOE, trade-id = bg54nm, product=ABC) Fails condition (f)
# Trade8: (date = '2022-03-15', time=9:02:23, type=Electronic, qty=-200, strike=1550, expiry='2022-04-28', kind=P
#         exchange=CBOE, trade-id = 6y7finm, product=ABC) [Fails condition (g)]

TRADE_FIELDS = {'date', 'time', 'type', 'qty', 'strike', 'expiry', 'kind', 'exchange', 'trade-id', 'product', 'side'}
class Trade:
    def __init__(self, input):
        self.data = {field: None for field in TRADE_FIELDS}
        try:
            for field_data in input:
                field_name, field_data = field_data.split('=')
                self.data[field_name] = field_data
            self.datetime = datetime.datetime.strptime(self.data['date'] + '/' + self.data['time'], "'%Y-%m-%d'/%H:%M:%S")
            self.data['qty'] = int(self.data['qty'])
            if self.data['exchange'] == 'CBOE':
                self.data['side'] = 'B' if int(self.data['qty']) > 0 else 'S'
                self.data['qty'] = abs(int(self.data['qty']))
        except:
            pass

def running_pair(broker: Trade, electronic: Trade):
    # b
    if not electronic.datetime <= broker.datetime <= electronic.datetime + datetime.timedelta(seconds=60):
        print('b')
        return False
    # c
    if broker.data['product'] != electronic.data['product']:
        print('c')
        return False
    # d
    if broker.data['kind'] != electronic.data['kind']:
        print('d')
        return False
    # e
    if broker.data['side'] != electronic.data['side']:
        print('e')
        return False
    # f
    if broker.data['expiry'] != electronic.data['expiry']:
        print('f')
        return False
    # g
    if broker.data['strike'] != electronic.data['strike']:
        print('g')
        return False
    return True



class Solution:
    def __init__(self):
        self.broker_feeds = []
        self.electronic_feeds = []

    def process_raw_trade(self, raw_trade: List):
        trade = Trade(raw_trade)
        if trade.data['type'] == "Broker":
            self.broker_feeds.append(trade)
        elif trade.data['type'] == "Electronic":
            self.electronic_feeds.append(trade)
        print(trade.data)
    def run(self):
        res = []
        for broker in self.broker_feeds:
            for electronic in self.electronic_feeds:
                if running_pair(broker, electronic):
                    res.append((broker, electronic))

        return [(broker.data['trade-id'], electronic.data['trade-id'])
                for broker, electronic in sorted(res, key=lambda x: x[1].datetime)]




if __name__ == '__main__':
    rows = []
    rows.append(
        "date = '2022-03-15', time=9:01:00, type=Broker, qty=-500, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE,trade-id = 737acm, product=ABC")
    # rows.append(
    #     "date = '2022-03-15', time=9:00:24, type=Electronic, qty=-200, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE, trade-id = w6c229, product=ABC")
    # rows.append(
    #     "date = '2022-03-15', time=9:03:45, type=Electronic, qty=-100, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE, trade-id = tssrin, product=ABC")
    # rows.append(
    #     "date = '2022-03-15', time=9:00:53, type=Electronic, qty=-500, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE, trade-id = lk451a, product=XYZ")
    # rows.append(
    #     "date = '2022-03-15', time=9:00:05, type=Electronic, qty=-350, strike=1500, expiry='2022-04-28', kind=C, exchange=CBOE, trade-id = 9numpr, product=ABC")
    # rows.append(
    #     "date = '2022-03-15', time=9:00:35, type=Electronic, qty=200, strike=1500, expiry='2022-04-28', kind=P, exchange=CBOE, trade-id = 922v3g, product=ABC")
    # rows.append(
    #     "date = '2022-03-15', time=9:00:47, type=Electronic, qty=-150, strike=1500, expiry='2022-04-21', kind=P, exchange=CBOE, trade-id = bg54nm, product=ABC")
    rows.append(
        "date = '2022-03-15', time=9:02:23, type=Electronic, qty=-200, strike=1550, expiry='2022-04-28', kind=P, exchange=CBOE, trade-id = 6y7finm, product=ABC")
    solution = Solution()
    for row in rows:
        raw_trade = list(row.strip().replace(" ", "").split(","))
        solution.process_raw_trade(raw_trade)


    print(solution.run())