from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.pipeline import make_pipeline
df = pd.read_csv("FIN.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df[(df['Date'] > '01/01/2020')]
df.set_index('Date', inplace=True)
df = df.reset_index(drop=True)

net_goals = df.HG - df.AG
f = df.HG == 0
h = df.AG == 0
'''Feature engineering; columns like points home, points away, home team goal difference, away team goal difference,
failed to score (Home and Away), both teams scored (Home and Away), points per game(Home and Away), 
average goals scored(Home and Away), average goals conceded(Home and Away) etc'''
df['Over1.5'] = np.where((df.HG + df.AG) >= 2, 1, 0)
df['Under1.5'] = np.where((df.HG + df.AG) <= 1, 1, 0)
df['ph'] = [3 if ng>0 else 1 if ng==0 else 0 for ng in net_goals]
df['pa'] = [3 if ng<0 else 1 if ng==0 else 0 for ng in net_goals]
df['hgd'] = df.HG - df.AG
df['agd'] = df.AG - df.HG
df['ftsH'] = [1 if i==0 else 0 for i in f]
df['ftsA'] = [1 if i==0 else 0 for i in h]
df['btsH'] = [1 if i==1 else 0 for i in f]
df['btsA'] = [1 if i==1 else 0 for i in h]
df['ppgh'] = df.groupby('Home')['ph'].apply(lambda x: x.rolling(5, closed='left').mean())
df['ppga'] = df.groupby('Away')['pa'].apply(lambda x: x.rolling(5, closed='left').mean())
df['avgh'] = df.groupby('Home')['HG'].apply(lambda x: x.rolling(5, closed='left').mean())
df['avga'] = df.groupby('Away')['AG'].apply(lambda x: x.rolling(5, closed='left').mean())
df['avcgh'] = df.groupby('Home')['AG'].apply(lambda x: x.rolling(5, closed='left').mean())
df['avcga'] = df.groupby('Away')['HG'].apply(lambda x: x.rolling(5, closed='left').mean())
df['avhgd'] = df.groupby('Home')['hgd'].apply(lambda x: x.cumsum().shift())
df['avagd'] = df.groupby('Away')['agd'].apply(lambda x: x.cumsum().shift())
df['ftsh'] = df.groupby('Home')['ftsH'].apply(lambda x: x.rolling(5, closed='left').mean())
df['ftsa'] = df.groupby('Away')['ftsA'].apply(lambda x: x.rolling(5, closed='left').mean())
df['btsh'] = df.groupby('Home')['btsH'].apply(lambda x: x.rolling(5, closed='left').mean())
df['btsa'] = df.groupby('Away')['btsA'].apply(lambda x: x.rolling(5, closed='left').mean())
df['o1.5H'] = df.groupby('Home')['Over1.5'].apply(lambda x: x.rolling(5, closed='left').mean())
df['o1.5A'] = df.groupby('Away')['Over1.5'].apply(lambda x: x.rolling(5, closed='left').mean())
df['u1.5H'] = df.groupby('Home')['Under1.5'].apply(lambda x: x.rolling(5, closed='left').mean())
df['u1.5A'] = df.groupby('Away')['Under1.5'].apply(lambda x: x.rolling(5, closed='left').mean())
df['XGh'] = (df.avgh + df.avcgh) / 2
df['XGa'] = (df.avga + df.avcga) / 2
#poisson distribution function for both teams scored
def btts_prob(a, b):   
	home_goals_vector = poisson(a).pmf(np.arange(0, 26))
	away_goals_vector = poisson(b).pmf(np.arange(0, 26))
	m = np.outer(home_goals_vector, away_goals_vector)
	s = np.sum(m[1:, 1:])
	return s
#poisson distribution function for total goals scored
def over(a, b):   
	home_goals_vector = poisson(a).pmf(np.arange(0, 26))
	away_goals_vector = poisson(b).pmf(np.arange(0, 26))
	m = np.outer(home_goals_vector, away_goals_vector)
	q = np.sum(m[0, 2:]) + np.sum(m[1, 1:]) + np.sum(m[2:])
	return q
df['bttsprob'] = df.apply(lambda x: btts_prob(x.XGh, x.XGa), axis=1)
df['btsn'] = 1 - df.bttsprob
df['over15'] = df.apply(lambda x: over(x.XGh, x.XGa), axis=1)
df['under15'] = 1 - df.over15      
df['BTTS'] = np.where((df.HG > 0) & (df.AG > 0), 1, 0)
df.dropna(inplace=True)
#df = df.sort_values('Date')

x = df[['Home', 'Away', 'ppga', 'ppgh', 'avgh',
	    'avga', 'avcgh', 'avcga', 'ftsh', 'ftsa', 'avhgd', 'avagd',
	    'btsh', 'btsa', 'o1.5H', 'o1.5A', 'u1.5H', 
	    'u1.5A', 'over15', 'under15', 'bttsprob', 'btsn']].values
y = df['Over1.5']
print(y.value_counts(normalize=True))
#print(df.to_string())
model = LogisticRegression(C=1.0, max_iter=4000)
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer(
    [('OHR', OneHotEncoder(sparse=False), [0,1])],
    remainder = 'passthrough'
    )
x = ohe.fit_transform(x)
ts = TimeSeriesSplit()
'''Cross validation, using TimeSeriesSplit() is recommended when evaluating
time-series data'''
print(cross_val_score(model, x, y, scoring='accuracy', cv=ts).mean())
model.fit(x, y)
print(model.score(x, y))
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_pred = model.predict(x_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("f1 score:", f1_score(y_test, y_pred))
print("auc score:", roc_auc_score(y_test, y_pred))

# Predicted probability of 2 classes
y_pred_new_threshold = (model.predict_proba(x_test)[:,1]>=0.7).astype(int)
print("Accuracy Threshold of 60%:", round(accuracy_score(y_test, y_pred_new_threshold), 3))

# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
df = df.drop(['Country', 'Season', 'Time', 'League',
              'MaxH', 'MaxD', 'MaxA', 'AvgH',
              'AvgD', 'AvgA', 'ph', 'pa', 'PA', 'PH', 'PD'], axis=1)
df['pred'] = model.predict(x)
print(model.predict_proba(x))
        