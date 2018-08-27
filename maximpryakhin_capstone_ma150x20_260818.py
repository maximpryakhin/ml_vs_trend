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



from sklearn import ensemble

import quandl

#Class retu calculates performance indicators of the trading strategy, it get the values
#of ['Win%','Win/loss','Mean trade ret','Sharpe','Cons losers','Max dd','Cagr','Lake','GPR']
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
#max drawdown function        
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


#get function produces a dataframe of prices for the given period and stochastic of same
def get(stock):
#Stochastic(14) function input prices    
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

#API quandl key 
 quandl.ApiConfig.api_key = "Y4Lhx9iJQ-dsf9xHUbxG"

#dates 
 start = datetime.datetime(2009,1,1)
 end = datetime.datetime(2018,6,1)

#Produce dataframe of prices with similar date indexes joining individual Quandl price vectors
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

#Simple moving 150 and 20 average function input prices and stock names 
def ma(prices,stock):
  
 
  ma=prices.rolling(window=20).mean()
    
  ma1=prices.rolling(window=150).mean()   
  return ma,ma1

# function finds the total return of the strategy for the validation period and returns a dictionary   
def find(prices,ma,ma1, stock):
 
 sett=prices.copy()
 
 

 for s in stock:
  pos=0
  pnl =[]
  portf=[1]
  pr=0
  
#loop for 150x20 crossover signals and trade them accumulating profits in the return index portf  
  for k in range(0,len(prices)):
    portf.append(portf[-1]+pos*(prices[s][k]-prices[s][k-1]))
     
    
   
    if ma[s][k-1]>ma1[s][k-1] and ma[s][k]<ma1[s][k]: 
      
        #print('close lng',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=-1/prices[s][k]
        pr=prices[s][k]
        #print('open sh',k,'-',prices[s][k])
          
    if ma[s][k-1]<ma1[s][k-1] and ma[s][k]>ma1[s][k]:
        #print('close sh',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=1/prices[s][k]
        pr=prices[s][k]
        #print('open lng',k,'-',prices[s][k])
     
    
  pnl.append(-pos*(pr-prices[s][k]))   
  sett[s]=portf[1:]
 best=dict(sett.iloc[-1,:])
 return best   



#main
if __name__ == "__main__":
#stock names list 
 stock=['AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','XOM','GE','GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','UTX','UNH','VZ','WMT']
#get prices 
 prices,sto=get(stock)
#get 20x150 MA
 ma,ma1=ma(prices,stock)
#get the validation return
 st = find(prices[:1201],ma[:1201],ma1[:1201],stock)
 ind=pd.DataFrame() 
 nms=[]
 sett=prices[1201:].copy()
 info =pd.DataFrame()

#loop THROUGH STOCK NAMES
 for s in stock:
  pos=0
  pnl =[]
  portf=[1]
  pr=0
  
  ind[s]=prices[s][1201:]/prices[s][1201]
#if validation return is below 25% DROP THE STOCK  
  if st[s]<1.3:
      
      continue
  nms.append(s)
  
#loop for given stock for signal crossovers for the test period and trade  
  for k in range(1201,len(prices)):
    portf.append(portf[-1]+pos*(prices[s][k]-prices[s][k-1]))
     
    
     
    if ma[s][k-1]<ma1[s][k-1] and ma[s][k]>ma1[s][k]: 
      
        #print('close s',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=1/prices[s][k]
        pr=prices[s][k]
        #print('open l',k,'-',prices[s][k])
          
    if ma[s][k-1]>ma1[s][k-1] and ma[s][k]<ma1[s][k]:
        #print('close l',k,'-',prices[s][k])
        pnl.append(-pos*(pr-prices[s][k]))
        pos=-1/prices[s][k]
        pr=prices[s][k]
        #print('open s',k,'-',prices[s][k])
     
    
  pnl.append(-pos*(pr-prices[s][k]))
     
  sett[s]=portf[1:]
  
#get strategy stock return performance indicators and print
  c=retu(portf[1:],pnl)
  
  info[s] =c.set
  
  
  print(s, ' stats')
  x=retu(np.asarray(portf[1:]),pnl)    
  x.prt()

#plot strategy 1usd value  
 (sett[nms]).mean(axis=1).plot(label='algo')
#get performance dataframe and print
 info.index=c.nms
 info['mean']=info.mean(axis=1)
 
 ind.mean(axis=1).plot(label='i')
 plt.legend(loc='upper left')
 plt.title('1 usd strategy return performance')
 plt.show() 
 
 print(info)   
      
     


