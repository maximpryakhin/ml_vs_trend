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
from sklearn import linear_model,datasets, svm, cross_validation, tree, preprocessing, metrics

import datetime



from sklearn import ensemble

import quandl
from sklearn.metrics import mean_squared_error

#Class retu calculates performance indicators of the trading strategy, it get the values
#of ['Win%','Win/loss','Mean trade ret','Sharpe','Cons losers','Max dd','Cagr','Lake','GPR']
class retu():
    
    def __init__(self, val,pl):
        self.val = val
        self.pl = pl
        df=pd.DataFrame(val)
        self.lake = (df.expanding(min_periods=1).max()-df).sum()/df.sum()
        self.gpr= (sum(i for i in pl if i >=0)/-sum(i for i in pl if i <0))  
        self.wr=(sum(1 for i in pl if i >0))/len(pl)  
        self.wl=  (sum(1 for i in pl if i >0))/(sum(1 for i in pl if i <0))
       
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
        
        self.maxcl=max(z)/len(c)
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

#Function rsi() gets the values of the RSI(10) for the price set
def rsi(jack):
 up, down = jack.diff().dropna(), jack.diff().dropna()
 up[up < 0] = 0
 down[down > 0] = 0


# Calculate the SMA
 roll_up = up.rolling(window=10,center=False).mean()
 roll_down = down.rolling(window=10,center=False).mean().abs()

# Calculate the RSI based on SMA

 RS = roll_up / roll_down
 RSI = 100.0 - (100.0 / (1.0 + RS))
 RSI=RSI.fillna(method='ffill')
 return RSI
 
#Macd() gets the values for MACD(5,10)
def MACD(df):
    
    
    df1 = df.ewm(span=5,min_periods=12,adjust=False).mean()
    df2 = df.ewm(span=10,min_periods=26,adjust=False).mean()
    df3 = (df2 - df1)
    dfs = df3.ewm(span=9,min_periods=5,adjust=False).mean()
    dfc = df3 - dfs
    return df3[1:]/df[1:]


#get() gets the values for the analyzed price set for the given period, stochastic(10) values
def get(stock):
    
#stochastic function    
 def stoch(df):
 

  #Create the "L14" column in the DataFrame
   df['L14'] = df['Low'].rolling(window=10).min()

#Create the "H14" column in the DataFrame
   df['H14'] = df['High'].rolling(window=10).max()

#Create the "%K" column in the DataFrame
   df['K'] = 100*((df['Close'] - df['L14']) / (df['H14'] - df['L14']) )

#Create the "%D" column in the DataFrame
   df['D'] = df['K'].rolling(window=3).mean()

   return df['K'][1:]  
#quandl API key 
 quandl.ApiConfig.api_key = "Y4Lhx9iJQ-dsf9xHUbxG"
 
#Dates
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

#returns
 ret = np.log(jack).diff().dropna()
 
 r5=ret.rolling(window=5).mean()
 return jack,sto,ret,r5
 
    
#MAIN
if __name__ == "__main__":

#list of stocks    
 stock=['AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','XOM','GE','GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','UTX','UNH','VZ','WMT']
#get prices, stochastic,returns,mean 5d returns
 jack,sto,ret,r5=get(stock)

 ind=pd.DataFrame() 
 nms=[]
#get Macd, RSI, and their differences 
 M=MACD(jack)
 RSI=rsi(jack)
 RSI1=RSI.diff()
 Mc=M.diff()
 stod=sto.diff()
 
#predict dataframe 
 predict_gb=ret[1201:].copy()
 predict_tree=ret[1201:].copy()
 predict_lasso=ret[1201:].copy()
 size=ret[1201:].copy()


 anal=pd.DataFrame()
 
#for loop loops through stocks to train and predict the returns for a given stock 
 for s in stock:
#GB parameters  
  params = {'n_estimators': 300, 'max_depth': 7, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
#input train array 
  y=np.asarray((ret[s][31:1201].copy()))
  q1=np.asarray(r5[s][30:1200])
  q2=np.asarray(M[s][30:1200])
 
  q3=np.asarray(RSI[s][30:1200])
  
  q4=np.asarray(RSI1[s][30:1200])
  q5=np.asarray(Mc[s][30:1200])
  q6=np.asarray(sto[s][30:1200])
  q7=np.asarray(stod[s][30:1200])
#scaling train array  
  X=np.stack((q1,q2,q3,q4,q5,q6,q7),axis=-1)
  sc = preprocessing.MinMaxScaler(feature_range =(0, 1))
  x = sc.fit_transform(X)

#initialize and fit algorithms DT,GB,Lasso
  clf = tree.DecisionTreeRegressor()
  clf1=ensemble.GradientBoostingRegressor(**params)
  clf3=linear_model.Lasso()
  clf3=clf3.fit(x,y)  
  clf.fit(x,y)
  clf1.fit(x,y)
  
#test prediction array  
  y=np.asarray((ret[s][1201:].copy()))
  q1=np.asarray(r5[s][1200:-1])
  q2=np.asarray(M[s][1200:-1])
 
  q3=np.asarray(RSI[s][1200:-1])
  
  q4=np.asarray(RSI1[s][1200:-1])
  q5=np.asarray(Mc[s][1200:-1])
  q6=np.asarray(sto[s][1200:-1])
  q7=np.asarray(stod[s][1200:-1])
  
  X=np.stack((q1,q2,q3,q4,q5,q6,q7),axis=-1)
#scale test array
  sc = preprocessing.MinMaxScaler(feature_range =(0, 1))
  x = sc.fit_transform(X)
#predict returns  
  pred1 = (clf.predict(x))
  pred2 = (clf1.predict(x))
  pred3=(clf3.predict(x))
  predict_tree[s]=pred1
  predict_gb[s]=pred2
  predict_lasso[s]=pred3
  
#MSE of predictions, input into  
  a =mean_squared_error(ret[s][1201:], predict_tree[s])
  b =mean_squared_error(ret[s][1201:], predict_gb[s])
  c =mean_squared_error(ret[s][1201:], predict_lasso[s])
 
  anal[s]=[a,b,c]
    
  
#print MSE table    
 anal['mean']=anal.mean(axis=1)
 anal.index=['DT','GB','LASSO'] 
 print(anal)

#TRADE DT algorithm 3 top predictions daily, get 1usd porfolio value retun cumsum 
 pos=predict_tree.abs().stack().groupby(level=0).nlargest(3).unstack().reset_index(level=1, drop=True).reindex(columns=predict_tree.columns)
 pos[~pos.isnull()]=0.3333
 dirr=predict_tree.copy()
 dirr[dirr>0]=1
 dirr[dirr<=0]=-1

 port=(ret[1201:]*pos*dirr).sum(axis=1).cumsum()

#TRADE GB algorithm 3 top predictions daily, get 1usd porfolio value return cumsum
 pos1=predict_gb.abs().stack().groupby(level=0).nlargest(3).unstack().reset_index(level=1, drop=True).reindex(columns=predict_gb.columns)
 pos1[~pos1.isnull()]=0.3333
 dirr1=predict_gb.copy()
 dirr1[dirr1>0]=1
 dirr1[dirr1<=0]=-1
 port1=(ret[1201:]*pos1*dirr1).sum(axis=1).cumsum()

#TRADE LASSO algorithm 3 top predictions daily, get 1usd porfolio value return cumsum
 pos2=predict_lasso.abs().stack().groupby(level=0).nlargest(3).unstack().reset_index(level=1, drop=True).reindex(columns=predict_lasso.columns)
 pos2[~pos2.isnull()]=0.3333
 dirr2=predict_lasso.copy()
 dirr2[dirr2>0]=1
 dirr2[dirr2<=0]=-1

 port2=(ret[1201:]*pos2*dirr2).sum(axis=1).cumsum()


#portfolio values plot vs. benchmark index plot
 pp=np.exp(port)
 pp1=np.exp(port1)
 pp2=np.exp(port2)
 pp.plot(label='tree')
 pp1.plot(label='grad')
 pp2.plot(label='lasso')
 po=(ret[1201:]).mean(axis=1).cumsum()
 p=np.exp(po)
 p.plot(label='i')
 plt.legend(loc='upper left')
 plt.title('1 usd strategy return performance')
 plt.show()

#input data into retu() class, get performance indicators, put into table nd print
 p1=np.asarray(pp)       
 p2=np.asarray(pp1)
 p3=np.asarray(pp2)

 pnl1=p1[1:]-p1[:-1]
 pnl2=p2[1:]-p2[:-1]
 pnl3=p3[1:]-p3[:-1]
 
 e =pd.DataFrame()

 c=retu(p1,pnl1)
 cc=retu(p2,pnl2)
 ccc=retu(p3,pnl3)
 cccc=retu(list(p),list(p.diff().dropna()))

 e['tree']=c.set
 e['gb']=cc.set
 e['lasso']=ccc.set
 e['index']=cccc.set
 e.index=c.nms
 print(e)
 