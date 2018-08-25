# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:02:26 2018

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:03:42 2018

@author: User
"""

import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics

import datetime

#import tensorflow as tf
#from tensorflow.contrib import skflow

from sklearn import ensemble
#GET DATA
import quandl

class retu():
    
    def __init__(self, val,pl):
        
                
        self.val = val
        self.pl = pl
        df=pd.DataFrame(val)
        self.lake = (df.expanding(min_periods=1).max()-df).sum()/df.sum()
        try:
         self.gpr= (sum(i for i in pl if i >=0)/-sum(i for i in pl if i <0))  
        except:
         self.gpr=5   
        try:
         self.wl= (sum(1 for i in pl if i >0))/(sum(1 for i in pl if i <0))
        except:
         
         self.wl=100
         
        self.wr=(sum(1 for i in pl if i >0))/len(pl)  
        
        self.avg=sum(pl)/len(pl)
        
        def max_dd(ser):
          max2here = ser.expanding(min_periods=1).max()
          dd2here = ser - max2here
          e=dd2here.values.tolist()
          r=min(e)
          price = max2here.iloc[e.index(r)]
          return dd2here.min()/price
      
        self.dd=float(max_dd(df))
        
        self.sharpe = (np.log(df).diff().dropna().mean())*(252**0.5)/np.log(df).diff().dropna().std()
        self.cagr=np.log(df).diff().dropna().mean()*252
        c=c=[1 if i<0 else 0 for i in pl]
        
        import itertools
        z=[[x[0],len(list(x[1]))] for x in itertools.groupby(c)]
        z=[i[1] for i in z if i[0]==1]
        try:
         self.maxcl=max(z)/len(c)
        except:
         self.maxcl=0
        self.set=[float(self.wr),float(self.wl),float(self.avg),float(self.sharpe),float(self.maxcl),float(self.dd),float(self.cagr),float(self.lake),float(self.gpr)] 
        self.nms=['Win%','Win/loss','Mean trade ret','Sharpe','Cons losers','Max dd','Cagr','Lake','GPR']
    def prt(self):
         
         print('Win rate: ',float(self.wr))
         print('Win/loss rate: ',float(self.wl))
         print('Average deal: ',float(self.avg))
         print('Sharpe: ',float(self.sharpe))  
         print('Consec losers %:',self.maxcl)
         print('Max drwdwn: ',float(self.dd))
         print('Cagr: ',float(self.cagr))
         print('Lake: ',float(self.lake))
         print('GPR: ',float(self.gpr))


         
def rsi(jack):
 up, down = jack.diff().dropna(), jack.diff().dropna()
 up[up < 0] = 0
 down[down > 0] = 0


# Calculate the SMA
 roll_up = up.rolling(window=21,center=False).mean()
 roll_down = down.rolling(window=21,center=False).mean().abs()

# Calculate the RSI based on SMA

 RS = roll_up / roll_down
 RSI = 100.0 - (100.0 / (1.0 + RS))
 RSI=RSI.fillna(method='ffill')
 return RSI
 

def MACD(df):
    
    
    df1 = df.ewm(span=12,min_periods=12,adjust=False).mean()
    df2 = df.ewm(span=26,min_periods=26,adjust=False).mean()
    df3 = (df2 - df1)
    dfs = df3.ewm(span=9,min_periods=5,adjust=False).mean()
    dfc = df3 - dfs
    return dfc[1:]









def get(stock):
 def stoch(df):
 

  #Create the "L14" column in the DataFrame
   df['L14'] = df['Low'].rolling(window=14).min()

#Create the "H14" column in the DataFrame
   df['H14'] = df['High'].rolling(window=14).max()

#Create the "%K" column in the DataFrame
   df['K'] = 100*((df['Close'] - df['L14']) / (df['H14'] - df['L14']) )

#Create the "%D" column in the DataFrame
   df['D'] = df['K'].rolling(window=3).mean()

   return df['K'][1:]  
 
 quandl.ApiConfig.api_key = "Y4Lhx9iJQ-dsf9xHUbxG"
 
 #for k in stock:
 start = datetime.datetime(2009,1,1)
 end = datetime.datetime(2018,6,1)
 #df=web.DataReader(stock, 'google', start, end)['Close']
 jack = pd.DataFrame(quandl.get("WIKI/"+stock[0], start_date=start, end_date=end))
 sto=pd.DataFrame(stoch(jack))
 sto.columns=[stock[0]]
 jack=pd.DataFrame(jack['Adj. Close'])
 jack.columns=[stock[0]]
 print(stock[0],' ',len(jack))

 for k in stock[1:]:
  price = pd.DataFrame(quandl.get("WIKI/"+k, start_date=start, end_date=end))    
  s=pd.DataFrame(stoch(price))
  s.columns=[k]
  price=pd.DataFrame(price['Adj. Close'])
  price.columns=[k]
 
  if len(price)/len(jack)<0.95:
     continue
  jack=pd.merge(jack, price, left_index=True, right_index=True)
  sto=pd.merge(sto, s, left_index=True, right_index=True)
 
  print(k,' ',len(price))   


 prices=jack[1:]
 
 return prices,sto
 
def ma(prices,stock):
  
 
  ma=prices.rolling(window=20).mean()
    
  ma1=prices.rolling(window=150).mean()   
  return ma,ma1
    
def find(prices,ma,ma1, stock):
 
 sett=prices.copy()
 
 

 for s in stock:
  pos=0
  pnl =[]
  portf=[1]
  pr=0
  
  
  for k in range(0,len(prices)):
    portf.append(portf[-1]+pos*(prices[s][k]-prices[s][k-1]))
     
    
   
    if ma[s][k-1]>ma1[s][k-1] and ma[s][k]<ma1[s][k]: 
      
        print('close shrt',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=1/prices[s][k]
        pr=prices[s][k]
        print('open long',k,'-',prices[s][k])
          
    if ma[s][k-1]<ma1[s][k-1] and ma[s][k]>ma1[s][k]:
        print('close long',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=-1/prices[s][k]
        pr=prices[s][k]
        print('open short',k,'-',prices[s][k])
     
    
  pnl.append(-pos*(pr-prices[s][k]))   
  sett[s]=portf[1:]
 best=dict(sett.iloc[-1,:])
 return best   




if __name__ == "__main__":
 
 stock=['AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','XOM','GE','GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','UTX','UNH','VZ','WMT']
 prices,sto=get(stock)
 ma,ma1=ma(prices,stock)
 st = find(prices[:1201],ma[:1201],ma1[:1201],stock)
 ind=pd.DataFrame() 
 nms=[]
 sett=prices[1201:].copy()
 info =pd.DataFrame()
 pnlo=[]
 for s in stock:
  pos=0
  pnl =[]
  portf=[1]
  pr=0
  
  ind[s]=prices[s][1201:]/prices[s][1201]
  
  if st[s]<1.25:
      
      continue
  nms.append(s)
  
  
  for k in range(1201,len(prices)):
    portf.append(portf[-1]+pos*(prices[s][k]-prices[s][k-1]))
     
    
     
    if ma[s][k-1]<ma1[s][k-1] and ma[s][k]>ma1[s][k]: 
      
        print('close shrt',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=1/prices[s][k]
        pr=prices[s][k]
        print('open long',k,'-',prices[s][k])
          
    if ma[s][k-1]>ma1[s][k-1] and ma[s][k]<ma1[s][k]:
        print('close long',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=-1/prices[s][k]
        pr=prices[s][k]
        print('open short',k,'-',prices[s][k])
     
    
  pnl.append(-pos*(pr-prices[s][k]))
  pnlo.append(pnl)   
  sett[s]=portf[1:]
  

  c=retu(portf[1:],pnl)
  
  info[s] =c.set
  
  
  print(s, ' stats')
  x=retu(np.asarray(portf[1:]),pnl)    
  x.prt()
 
 (sett[nms]).mean(axis=1).plot(label='algo')

 info.index=c.nms
 info['mean']=info.mean(axis=1)
 
 ind.mean(axis=1).plot(label='i')
 plt.legend(loc='upper left')
 plt.show() 
    
      
     


