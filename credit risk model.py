
import pandas as pd
import numpy as np
import scipy.stats as p
from scipy.stats import chi2_contingency as chi
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics

#Import the data
data = datasets[0]
data

#Create new datasets, those w/ delinquent loans and those without
datan = data[data['dq'] == 0]
datad = data[data['dq'] == 1]

#Create Max Credit Score column that takes the Max between the borrower(s) 
data['MaxCreditScore'] = data[['borrowercreditscore', 'coborrowercreditscore']].max(axis = 1)
data

#Experiemented with binary versions of dti and credit score, didn't end up using in analysis
def dtihigh(d):
  if d >= 38.0:
    return 1
  elif d < 38.0:
    return 0


data['highdti'] = [dtihigh(entry) for entry in data['dti']]

def cs_level(c):
  if c <= 735.0:
    x = 1
  else:
    x = 0
  return x
    
data['lowcs'] = [cs_level(entry) for entry in data['MaxCreditScore']]

#non-delinquent
datan.describe()

#delinquent
datad.describe()

crosstab_col1_col2 = pd.crosstab(index = data['MaxCreditScore'], columns = data['dq'])
chi_sq_result = p.chi2_contingency(crosstab_col1_col2)
p_value = chi_sq_result[1]
print('P Value Max Credit Score = ' + str(p_value))
print('Reject the null hypothesis -> there is a relationship btw the variables (5% sig level)')

fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=False)

#plots
ax[0].hist(datad['MaxCreditScore'], bins = 40)
ax[1].hist(datan['MaxCreditScore'], bins = 40)
fig.suptitle('Max Credit Score: Delinquent Loans vs. Non-Delinquent Loans', fontsize = 20)

#axis settings
ax[0].set_ylabel('Delinquent Loans', fontsize = 17)
ax[0].set_xlabel('MaxCreditScore', fontsize = 17)
ax[1].set_xlabel('MaxCreditScore', fontsize = 17)
ax[1].set_ylabel('Non-Delincuent Loans', fontsize = 17)

zips = zip(datan['MaxCreditScore'].describe(), datad['MaxCreditScore'].describe())
cols = ['Non-Delinquent', 'Delinquent']
stats = pd.DataFrame(zips, columns = cols)
stats.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
stats

crosstab_col1_col2 = pd.crosstab(index = data['lvratio'], columns = data['dq'])
chi_sq_result = p.chi2_contingency(crosstab_col1_col2)
p_value = chi_sq_result[1]
print('P Value Loan to Value Ratio = ' + str(p_value))
print('Reject the null hypothesis -> there is a relationship btw the variables (5% sig level)')

fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=False)

#plots
ax[0].hist(datad['lvratio'], bins = 40)
ax[1].hist(datan['lvratio'], bins = 40)
fig.suptitle('Loan to Value Ratio: Delinquent Loans vs. Non-Delinquent Loans', fontsize = 20)

#axis settings
ax[0].set_ylabel('Delinquent Loans', fontsize = 17)
ax[0].set_xlabel('Loan to Value Ratio', fontsize = 17)
ax[1].set_xlabel('Loan to Value Ratio', fontsize = 17)
ax[1].set_ylabel('Non-Delincuent Loans', fontsize = 17)

zips = zip(datan['lvratio'].describe(), datad['lvratio'].describe())
cols = ['Non-Delinquent', 'Delinquent']
stats = pd.DataFrame(zips, columns = cols)
stats.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
stats

crosstab_col1_col2 = pd.crosstab(index = data['dti'], columns = data['dq'])
chi_sq_result = p.chi2_contingency(crosstab_col1_col2)
p_value = chi_sq_result[1]
print('P Value DTI = ' + str(p_value))
print('Reject the null hypothesis -> there is a relationship btw the variables (5% sig level)')

fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=False)

#plots
ax[0].hist(datad['dti'], bins = 30)
ax[1].hist(datan['dti'], bins = 30)
fig.suptitle('DTI: Delinquent Loans vs. Non-Delinquent Loans', fontsize = 20)

#axis settings
ax[0].set_ylabel('Delinquent Loans', fontsize = 17)
ax[0].set_xlabel('DTI', fontsize = 17)
ax[1].set_xlabel('DTI', fontsize = 17)
ax[1].set_ylabel('Non-Delincuent Loans', fontsize = 17)

zips = zip(datan['dti'].describe(), datad['dti'].describe())
cols = ['Non-Delinquent', 'Delinquent']
stats = pd.DataFrame(zips, columns = cols)
stats.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
stats

crosstab_col1_col2 = pd.crosstab(index = data['originalunpaidbalance'], columns = data['dq'])
chi_sq_result = p.chi2_contingency(crosstab_col1_col2)
p_value = chi_sq_result[1]
print('P Value Original Unpaid Balance = ' + str(p_value))
print('Reject the null hypothesis -> there is a relationship btw the variables (5% sig level)')

fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=False)

#plots
ax[0].hist(datad['originalunpaidbalance'], bins = 40)
ax[1].hist(datan['originalunpaidbalance'], bins = 40)
fig.suptitle('Original Unpaid Balance: Delinquent Loans vs. Non-Delinquent Loans', fontsize = 20)

#axis settings
ax[0].set_ylabel('Delinquent Loans', fontsize = 17)
ax[0].set_xlabel('Original Unpaid Balance', fontsize = 17)
ax[1].set_xlabel('Original Unpaid Balance', fontsize = 17)
ax[1].set_ylabel('Non-Delincuent Loans', fontsize = 17)

zips = zip(datan['originalunpaidbalance'].describe(), datad['originalunpaidbalance'].describe())
cols = ['Non-Delinquent', 'Delinquent']
stats = pd.DataFrame(zips, columns = cols)
stats.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
stats

approve_model = data.loc[(data['MaxCreditScore'] >= 699) | ((data['dti'] <= 38) & (data['lvratio'] < 75)) | ((data['originalunpaidbalance']<=300000) & (data['dti'] <= 17) & (data['MaxCreditScore'] >= 660))]

#BASELINE: Total % of loans w/ DQ
baseline_model = (data[data['dq'] == 1].shape[0] * 100)/data.shape[0]

TotalDQ = (approve_model[approve_model['dq'] == 1].shape[0] * 100)/approve_model.shape[0]
TotalDQ

TotalApproved = (approve_model.shape[0] * 100)/data.shape[0]

print('Baseline Delinquent %: ' + str(baseline_model))
print('Model Delinquent %: ' + str(TotalDQ))
print('Model Approved %: ' + str(TotalApproved))

