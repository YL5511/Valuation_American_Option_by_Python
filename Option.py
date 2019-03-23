#!/usr/bin/env python
# coding: utf-8

# #  Preprocessing
import pandas as pd
from scipy.stats import norm
import numpy as np


#import data
path='../data/201801.csv'
df=pd.read_csv(path)

#sort data by 'StockCode' and 'TradingDate'
df.sort_values(['StockCode','TradingDate'],ascending=True,inplace=True)
stk_price=df[['TradingDate', 'StockCode','ClosePrice']]
stk_price.TradingDate=pd.to_datetime(stk_price.TradingDate)

# Val is a dataframe which will be used to store results: S, K, R, sigma, T
Val=pd.DataFrame()

# ## Spot Price $S_0$  and Strike Price $K$
df_opt=stk_price

#the close price at 2018-01-15 is the spot price
spotprice=df_opt[df_opt['TradingDate']=='2018-01-15'][['StockCode','ClosePrice']]
spotprice.set_index('StockCode', inplace=True)
spotprice.rename(columns={'ClosePrice':'S_0'},inplace=True)
Val=spotprice

# strike price equals to strike price
Val['K']=Val['S_0']
Val.head()

# Annual Volatility : sigma


#daily return
df_opt['d_rt']=df_opt.groupby('StockCode')['ClosePrice'].shift(0)/df_opt.groupby('StockCode')['ClosePrice'].shift(1)-1
df_opt.head(5)

#transfor the daily volatility to annually volatility
df_past=df_opt[df_opt['TradingDate']<'2018-01-16']
df_past[df_past['StockCode']==1]
Val['sigma']=df_past.groupby('StockCode')['d_rt'].std(ddof=1)*(250**0.5) #change daily volatility to annual volatility 
BSM=Val
Val.head()

# Valuation
def BinModel( S0, K, R, sigma, N, dt):
    payoff=[]
    U=np.exp(sigma*np.sqrt(dt))
    D=np.exp(-sigma*np.sqrt(dt))
    P=(np.exp(R*dt)-D)/(U-D)
    for i in range(N+1):
        payoff.append(np.maximum(S0*np.power(U,N-i)*np.power(D,i)-K,0))
    for n in reversed(range(N)):
        for i in range(n+1):
            payoff_not_early=np.exp(-R*dt)*(P*payoff[i]+(1-P)*payoff[i+1])
            payoff_early=np.maximum(S0*np.power(U,n-i)*np.power(D,i)-K,0)
            payoff[i]=np.maximum(payoff_early,payoff_not_early)
    return payoff[0]

R = 0.03 #risk free rate
T = 1/24 # half month
N=15
dt=T/N
Val['Price_Binmodel']=BinModel(Val['S_0'], Val['K'], R, Val['sigma'], N, dt)

Val=Val.dropna(axis=0,how='any')
Val.reset_index(level=0, inplace=True)
Val.head()

#Profit and Payoff
#exe calculate when is the optimal time to exercise the option
def exe(R,x):
    l_payoff=[]
    K=float(Val[Val['StockCode']==x]['K'])
    stockprice=list(df_opt[df_opt['StockCode']==x][df_opt['TradingDate']>'2018-01-15']['ClosePrice'])
    for i in range(len(stockprice)):
        payoff=np.maximum(stockprice[i]-K,0)
        l_payoff.append(payoff*np.exp(R*(T-i/356)))
        date=l_payoff.index(max(l_payoff))
    return [max(l_payoff),date]

stockcode=list(Val['StockCode'])
all_stock_payoff=[]
time_to_early_exe=[]

for x in stockcode:
    all_stock_payoff.append(exe(R,x)[0])
    time_to_early_exe.append(exe(R,x)[1])
    
Val['Payoff']=all_stock_payoff
Val['Time_early_exercise']=time_to_early_exe
Val['Profit']=Val['Payoff']-Val['Price_Binmodel']*np.exp(R*T) 
Val.head()
Val.to_csv('../data/3_Option_Valuation_Binmodel.csv',index=False)


# # Valuation By BSM model
#Opt_Val is function which calculates option price by BSM model
def Opt_Val_BSM(S, K, R, sigma, T):
    d_1=(np.log(S/K)+(R+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d_2=d_1-sigma*np.sqrt(T)
    C=norm.cdf(d_1)*S-norm.cdf(d_2)*K*np.exp(-R*T)
    return C

BSM['Price']=Opt_Val_BSM(BSM['S_0'], BSM['K'], R, BSM['sigma'], T)
BSM.head()


# ## Payoff and Profit
#St: is the close price of stock when option expire
expire_price=df_opt[df_opt['TradingDate']=='2018-01-31'][['StockCode','ClosePrice']]
expire_price.set_index('StockCode', inplace=True)
expire_price.rename(columns={'ClosePrice':'S_T'}, inplace=True)
BSM= pd.concat([BSM, expire_price], axis=1)
BSM.head(10)

#If St>S0, 'Exercise' equals True, that is, option will be exercised. 
BSM['Exercise ']=BSM['S_T']>BSM['S_0']
BSM.head()

#payoff= st-s0
BSM['Payoff']=np.maximum(0,BSM['S_T']-BSM['S_0'])
#profit = payoff-price with accumulate interest 
BSM['Profit']=BSM['Payoff']-BSM['Price']*np.exp(R*T)  
BSM=BSM.dropna(axis=0, how='any')
BSM.head()

#export 
BSM.to_csv('../data/3_Option_Valuation_BSM.csv')

