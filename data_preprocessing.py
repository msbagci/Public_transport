# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
import numpy as np
import warnings

# Hide warning messages
warnings.filterwarnings("ignore")

# In the csv file, columns are separated with ";"
df = pd.read_csv(f"ua_passengers_1995-2020.csv",delimiter=";")

# Converts the DataFrame from wide format to long format (unpivoting).
dff = pd.melt(frame=df,id_vars="year",value_vars=df.drop("year",axis=1).columns)

# Change in the number of uses of all means of transportation by year (trend chart)
plt.figure(figsize=(9,7))
sns.lineplot(df,x=dff["year"],y=dff["value"],hue=dff["variable"],marker="o")
plt.savefig("lineplot.png")
plt.close()

# Change in the number of uses of all means of transportation by year (bar chart)
fig,ax = plt.subplots(8,1,figsize=(15,30))
for i,ax in zip(df.drop("year",axis=1).columns, ax.flatten()):
            sns.barplot(df,ax=ax,x=df["year"],y=df[i],color="skyblue")
            
plt.savefig("barplot.png")
plt.close()
# The year column is used as an index 
dff = df.set_index("year")

# It is chosen as the most used means of transportation every year
max_transport = dff.idxmax(axis=1)

# It shows how much each vehicle was used in which year
values = dff.max(axis=1)

# It gives the most used vehicle type and number of uses by year
years = []
uses = []
nou = []
for type,value,year in zip(max_transport,values,dff.index):
    years.append(year)
    uses.append(type)
    nou.append(value)
mupt = pd.DataFrame({"Year":years,"Max Used Public Transport":uses,"Number of Uses":nou})
print(mupt)
    
# Usage rates of public transport vehicle types by year (pie chart)
fig,axes = plt.subplots(8,3,figsize=(12,36))
explode = {}
for i in dff.columns:
        explode[i] = 0
for i,ax in zip(dff.index,axes.flatten()):
        explode[dff.loc[i].idxmax()] = 0.1
        graph = ax.pie(x=dff.loc[i],labels=dff.columns,autopct="%1.1f%%",explode=list(explode.values()))
        ax.set_title(f"Year {i} Statistics")
        explode[dff.loc[i].idxmax()] = 0
plt.savefig("pie_graphs.png")       
plt.close()

# Total annual usage of transportation vehicles
df["sum"] = df.drop("year",axis=1).sum(axis=1)

x = df[["year"]]
y = df[["sum"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=3)
lm = LinearRegression()

# Model fitting
model = lm.fit(x_train,y_train)

# Prediction is made with linear regression
y_pred = model.predict(x_test)

# Linear Regression r2_score
linear_r2 = r2_score(y_pred=y_pred,y_true=y_test)
print(linear_r2)

# Dummy variable for intercept shift and Dummy variable for trend change
df["2014_indicator"] = np.where(df["year"]>=2014,1,0)
df["trend_change"] = df["2014_indicator"] * (df["year"]-2014)
x = df[["year","2014_indicator","trend_change"]]
y = df[["sum"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=2)

# Model fitting
model = lm.fit(x_train,y_train)

# Prediction is made with piecewise regression
y_pred = model.predict(x_test)


# Piecewise regression r2_score
piecewise_r2 = r2_score(y_pred=y_pred,y_true=y_test)
print(piecewise_r2)

# Artifical Neural Network regression
model = MLPRegressor()
model.fit(x_train,y_train)

# Artifiacl Neural Network r2_score
model.score(x_train,y_train)
y_pred = model.predict(x_test)
ann_r2 = r2_score(y_pred=y_pred,y_true=y_test)
print(ann_r2)

# OLS regression
model = sm.OLS(y, x).fit()

# Summary table
print(model.summary())


