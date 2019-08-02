import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

np.random.seed(1)
plt.style.use('ggplot')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
id=test.Id


X=pd.concat((train.drop(['Id','SalePrice'],axis=1),test.drop(['Id'],axis=1)))
y=np.log1p(train.SalePrice)

num=X.select_dtypes(exclude='object')
cat=X.select_dtypes(include='object')

lab_ec=LabelEncoder()

for col in cat.columns :
    X[col].fillna('None',inplace=True)
    X[col]=lab_ec.fit_transform(X[col])

for col in num.columns:
    X[col].fillna(0,inplace=True)
    skew_value=skew(X[col])
    if skew_value>0.75 :
        X[col]=np.log1p(X[col])


train=X[:train.shape[0]]
test=X[train.shape[0]:]


from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from mlxtend.regressor import StackingRegressor

lr=LinearRegression()
r=Ridge()
rf=RandomForestRegressor()
gb=GradientBoostingRegressor()

model=StackingRegressor(regressors=[r,rf,gb],meta_regressor=lr)

model.fit(train,y)

y_pred=model.predict(test)
# print(np.sqrt(mean_squared_error(y,y_pred))*100)


out=pd.DataFrame()
out['Id']=id
out['SalePrice']=np.expm1(y_pred)

out.to_csv('Sales_price_3.csv',index=False)

