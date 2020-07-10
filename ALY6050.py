import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import chisquare
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.api import OLS
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt

path = 'HON.csv'
df = pd.read_csv(path)

# exponential smoothing forecast
alphas = [0.15, 0.35, 0.55, 0.75]
colors = ['lightblue', 'darkorange', 'green', 'red']
price = df.Close
res = []
plt.figure(figsize=(10,5))
price.plot()
for i in range(len(alphas)):
    fit = SimpleExpSmoothing(price).fit(smoothing_level=alphas[i], optimized=False)
    pred = fit.forecast(1)
    fit.fittedvalues.plot(marker='o', markersize=2, 
                          color=colors[i], alpha=0.4, label='alpha=%.2f'%alphas[i])
    mse = (fit.resid ** 2).mean()
    res.append([alphas[i], mse, pred.values[0]])
plt.legend()
print(res)

# adjusted exponential smoothing forecast
alpha = 0.75
betas = [0.15, 0.25, 0.45, 0.85]
res = []
plt.figure(figsize=(10,5))
price.plot()
for i in range(len(betas)):
    beta = betas[i]
    fit = Holt(price).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)
    pred = fit.forecast(1)
    fit.fittedvalues.plot(marker='o', markersize=2, 
                          color=colors[i], alpha=0.4, label='alpha=%.2f'%beta)
    mse = (fit.resid ** 2).mean()
    res.append([beta, mse, pred.values[0]])
plt.legend()
print(res)

# regression, price vs. periods
exog = sm.add_constant(df.Period.values.reshape(-1,1))
reg =  OLS(df.Close.values, exog)
re = reg.fit()
print(re.summary())
fitted = re.fittedvalues
resid = re.resid
print('MSE: %f'%(resid**2).mean())
# a
print('Coefficient of correlation: %f'%np.corrcoef(fitted, price.values)[0,1])
print('Coefficient of determination: %f'%re.rsquared)

# b
count, bins, _ = plt.hist(resid)

# c
std = np.std(resid)
p = [norm.cdf(i, loc=0, scale=std) for i in bins]
p = p[1:2] + [p[i+1]-p[i] for i in range(1,len(p)-2)] + [1-p[-1]]
f_exp = [i * sum(count) for i in p]
pvalue = chisquare(count, f_exp).pvalue
print('p value: %f'%pvalue)
if pvalue<0.05:
    print('reject the null hypothesis')
else:
    print('can not reject the null hypothesis')

# d
qqplot(resid, line='s')

# e
plt.scatter(range(len(resid)), resid)
plt.xlabel('time')
plt.ylabel('residuals')

# f
plt.scatter(price.values, np.abs(resid))
plt.xlabel('price')
plt.ylabel('residuals')

pred = re.predict([1, 125])[0]
print('predicted price at 4/16/2018: %f'%pred)