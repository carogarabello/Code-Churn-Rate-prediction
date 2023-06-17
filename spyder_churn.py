# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:58:34 2023

@author: ADMIN
"""
#%%
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


#%%
#Sklearn modules
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

#Statistics
from statistics import median

#%%
# **Reading the data**
df1 = pd.read_csv("data1.csv")

df2 = pd.read_csv("data2.csv")

df3 = pd.read_csv("data3.csv")

df4 = pd.merge(pd.merge(df1,df2,on='id'),df3, on='id')

print(df4) 

#%%

# Is used to get a concise summary of the DataFrame including the index dtype and columns, non-null values and memory usage
df4.info()

# Return a random sample of items from an axis of object
df4.sample(15)
#%%
# **Data Exploration**

# Returns a tuple (number of rows, number of columns)representing the dimensionality of the DataFrame
df4.shape
#%%
# Is used to get a concise summary of the DataFrame including the index dtype and columns, non-null values and memory usage
df4.info()

#%%
# This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.
df4.head()
#%%

# Generate descriptive statistics.
df4.describe()

#%%

# Create arrays with descriptions so we can see all in the variable ecplorer
describe = df4.describe()

#%%

# Returns description of the data in the DataFrame, now including the object
df4.describe(include="object")
describe_object = df4.describe(include="object")
#%%
# **Data Cleaning**

# Features' arrays
feature_list = list(df4.columns)

#%%

# Let's see all the labels that are object
df4.select_dtypes('O').info()

#%%

# Function to get count of missing values in each column
missing_values_df = df4.isna().sum()
missing_values_df

#%%

# Count the total number of missing values
print(df4.isnull().values)
print(type(df4.isnull().values))
print(df4.isnull().values.sum())  # there are 1130954 NaN
#%%
# Plot of percentage of missing values
plt.xticks(rotation='vertical')
sns.barplot(x=missing_values_df.index, y=missing_values_df)
plt.title('Percentage of missing values')
#%%

# Delete features with 50% or more missing values 
df4.drop(['activity_new', 'campaign_disc_ele', 'date_first_activ', 'forecast_base_bill_ele', 'forecast_base_bill_year', 'forecast_bill_12m', 'forecast_cons'], axis=1, inplace=True)
#%%
# Delete features that we do not consider relevant
df4.drop(['id', 'date_activ', 'date_end', 'price_date'], axis=1, inplace=True)
#%%

# The percentage of missing values on the whole dataset does not exceed 55%, so we decide to replace the missing values with mean, mode and median depending on the type and problems of the variables
#%%
# Aanalisis and data cleaning of each variable

# channel_sales
channel_sales = df4['channel_sales'].value_counts(dropna=False)
#%%
# Nan replacement with channel_sales mode
df4['channel_sales'] = df4['channel_sales'].fillna(df4['channel_sales'].mode()[0])

#%%
#cons_12m
cons_12m = df4['cons_12m'].value_counts(dropna=False)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['cons_12m'],grid=False, rot=45, fontsize=15)
#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[cons_12m].median()
    std = df4[cons_12m].std()
    outliers = ((df4[cons_12m] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[cons_12m].fillna(median, inplace=True)
    #%%
# Replace outliers with the median of cons_12m
#df4['cons_12m'] = df4['cons_12m'].fillna(df4['cons_12m'].median())
#%%
#cons_gas_12m
cons_gas_12m = df4['cons_gas_12m'].value_counts(dropna=False)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['cons_gas_12m'],grid=False, rot=45, fontsize=15)
#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[cons_gas_12m].median()
    std = df4[cons_gas_12m].std()
    outliers = ((df4[cons_gas_12m] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[cons_gas_12m].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['cons_gas_12m'] = df4['cons_gas_12m'].fillna(df4['cons_gas_12m'].median())
#%%
#cons_last_month
cons_last_month= df4['cons_last_month'].value_counts(dropna=False)

#%%
# Delete the negative number 
df4.drop(df4[df4['cons_last_month'] < 0].index, axis=0, inplace=True)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['cons_last_month'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[cons_last_month].median()
    std = df4[cons_last_month].std()
    outliers = ((df4[cons_last_month] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[cons_last_month].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['cons_last_month'] = df4['cons_last_month'].fillna(df4['cons_last_month'].median())
#%%
#date_modif_prod
date_modif_prod = df4['date_modif_prod'].value_counts(dropna=False)
#%%
# Nan replacement with date_modif_prod mode
df4['date_modif_prod'] = df4['date_modif_prod'].fillna(df4['date_modif_prod'].mode()[0])

#%%
# date_renewal
date_renewal = df4['date_renewal'].value_counts(dropna=False)
#%%
# Nan replacement with date_renewal mode
df4['date_renewal'] = df4['date_renewal'].fillna(df4['date_renewal'].mode()[0])
#%%
#forecast_cons_12m 
forecast_cons_12m  = df4['forecast_cons_12m'].value_counts(dropna=False)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['forecast_cons_12m'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[forecast_cons_12m].median()
    std = df4[forecast_cons_12m].std()
    outliers = ((df4[forecast_cons_12m] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[forecast_cons_12m].fillna(median, inplace=True)
# Replace outliers with the median of cons_12m
#df4['forecast_cons_12m'] = df4['forecast_cons_12m'].fillna(df4['forecast_cons_12m'].median())
#%%

#forecast_cons_year
forecast_cons_year = df4['forecast_cons_year'].value_counts(dropna=False)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['forecast_cons_year'],grid=False, rot=45, fontsize=15)
#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[forecast_cons_year].median()
    std = df4[forecast_cons_year].std()
    outliers = ((df4[forecast_cons_year] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[forecast_cons_year].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['forecast_cons_year'] = df4['forecast_cons_year'].fillna(df4['forecast_cons_year'].median())
#%%

#forecast_discount_energy
forecast_discount_energy = df4['forecast_discount_energy'].value_counts(dropna=False)
#%%
# Replace the missing values with its median
df4['forecast_discount_energy'] = df4['forecast_discount_energy'].fillna(df4['forecast_discount_energy'].median())
#%%
#forecast_meter_rent_12m
forecast_meter_rent_12m = df4['forecast_meter_rent_12m'].value_counts(dropna=False)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['forecast_meter_rent_12m'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[forecast_meter_rent_12m].median()
    std = df4[forecast_meter_rent_12m].std()
    outliers = ((df4[forecast_meter_rent_12m] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[forecast_meter_rent_12m].fillna(median, inplace=True)
#%%
#%%
# Replace outliers with the median of cons_12m
#df4['forecast_meter_rent_12m'] = df4['forecast_meter_rent_12m'].fillna(df4['forecast_meter_rent_12m'].median())
#%%

#forecast_price_energy_p1
forecast_price_energy_p1 = df4['forecast_price_energy_p1'].value_counts(dropna=False)
#%%
# Replace the missing values with its median
df4['forecast_price_energy_p1'] = df4['forecast_price_energy_p1'].fillna(df4['forecast_price_energy_p1'].median())
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['forecast_price_energy_p1'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[forecast_price_energy_p1].median()
    std = df4[forecast_price_energy_p1].std()
    outliers = ((df4[forecast_price_energy_p1] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[forecast_price_energy_p1].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['forecast_price_energy_p1'] = df4['forecast_price_energy_p1'].fillna(df4['forecast_price_energy_p1'].median())
#%%
#forecast_price_energy_p2
forecast_price_energy_p2 = df4['forecast_price_energy_p2'].value_counts(dropna=False)
#%%
# Replace the missing values with its median
df4['forecast_price_energy_p2'] = df4['forecast_price_energy_p2'].fillna(df4['forecast_price_energy_p2'].median())
#%%
#forecast_price_pow_p1
forecast_price_pow_p1 = df4['forecast_price_pow_p1'].value_counts(dropna=False)
#%%
# Delete the negative number 
df4.drop(df4[df4['forecast_price_pow_p1'] < 0].index, axis=0, inplace=True)
#%%
# Replace the missing values with its median
df4['forecast_price_pow_p1'] = df4['forecast_price_pow_p1'].fillna(df4['forecast_price_pow_p1'].median())
#%%
#has_gas
has_gas = df4['has_gas'].value_counts(dropna=False)
#%%
#imp_cons
imp_cons = df4['imp_cons'].value_counts(dropna=False)

#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['imp_cons'],grid=False, rot=45, fontsize=15)

#%%
# Replace outliers with the median of cons_12m
#df4['imp_cons'] = df4['imp_cons'].fillna(df4['imp_cons'].median())
#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[imp_cons].median()
    std = df4[imp_cons].std()
    outliers = ((df4[imp_cons] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[imp_cons].fillna(median, inplace=True)
    #%%
#margin_gross_pow_ele
margin_gross_pow_ele = df4['margin_gross_pow_ele'].value_counts(dropna=False)
#%%
# Delete the negative number 
df4.drop(df4[df4['margin_gross_pow_ele'] < 0].index, axis=0, inplace=True)
#%%
# Replace the missing values with its median
df4['margin_gross_pow_ele'] = df4['margin_gross_pow_ele'].fillna(df4['margin_gross_pow_ele'].median())
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['margin_gross_pow_ele'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[margin_gross_pow_ele].median()
    std = df4[margin_gross_pow_ele].std()
    outliers = ((df4[margin_gross_pow_ele] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[margin_gross_pow_ele].fillna(margin_gross_pow_ele, inplace=True)
    #%%
# Replace outliers with the median of cons_12m
#df4['margin_gross_pow_ele'] = df4['margin_gross_pow_ele'].fillna(df4['margin_gross_pow_ele'].median())
#%%
#margin_net_pow_ele
margin_net_pow_ele = df4['margin_net_pow_ele'].value_counts(dropna=False)
#%%
# Delete the negative number 
df4.drop(df4[df4['margin_net_pow_ele'] < 0].index, axis=0, inplace=True)
#%%
#Replace the missing values with its median
df4['margin_net_pow_ele'] = df4['margin_net_pow_ele'].fillna(df4['margin_net_pow_ele'].median())
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['margin_net_pow_ele'],grid=False, rot=45, fontsize=15)
#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[margin_net_pow_ele].median()
    std = df4[margin_net_pow_ele].std()
    outliers = ((df4[margin_net_pow_ele] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[margin_net_pow_ele].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['margin_net_pow_ele'] = df4['margin_net_pow_ele'].fillna(df4['margin_net_pow_ele'].median())
#%%
#nb_prod_act
nb_prod_act = df4['nb_prod_act'].value_counts(dropna=False)
#%%
#net_margin
net_margin = df4['net_margin'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['net_margin'] = df4['net_margin'].fillna(df4['net_margin'].median())
#%%
# Delete the negative number 
df4.drop(df4[df4['net_margin'] < 0].index, axis=0, inplace=True)
#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['net_margin'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[net_margin].median()
    std = df4[net_margin].std()
    outliers = ((df4[net_margin] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[net_margin].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['net_margin'] = df4['net_margin'].fillna(df4['net_margin'].median())
#%%
#num_years_antig
num_years_antig = df4['num_years_antig'].value_counts(dropna=False)
#%%
#origin_up
origin_up = df4['origin_up'].value_counts(dropna=False)
#%%
# Nan replacement with date_renewal mode
df4['origin_up'] = df4['origin_up'].fillna(df4['origin_up'].mode()[0])
#%%
#pow_max
pow_max = df4['pow_max'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['pow_max'] = df4['pow_max'].fillna(df4['pow_max'].median())

#%%
# From the boxplot we see that there are outlaier
boxplot = df4.boxplot(column=['pow_max'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[pow_max].median()
    std = df4[pow_max].std()
    outliers = ((df4[pow_max] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[pow_max].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['pow_max'] = df4['pow_max'].fillna(df4['pow_max'].median())
#%%
#price_p1_var
price_p1_var = df4['price_p1_var'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['price_p1_var'] = df4['price_p1_var'].fillna(df4['price_p1_var'].median())
#%%
# There are outliers
boxplot = df4.boxplot(column=['price_p1_var'],grid=False, rot=45, fontsize=15)

#%%
def replace_numerical_outliers(df4, column_name, z_thresh=3):
    median = df4[price_p1_var].median()
    std = df4[price_p1_var].std()
    outliers = ((df4[price_p1_var] - median).abs()) > z_thresh*std
    df4[outliers] = np.nan
    df4[price_p1_var].fillna(median, inplace=True)
#%%
# Replace outliers with the median of cons_12m
#df4['price_p1_var'] = df4['price_p1_var'].fillna(df4['price_p1_var'].median())
#%%
#price_p2_var
price_p2_var = df4['price_p2_var'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['price_p2_var'] = df4['price_p2_var'].fillna(df4['price_p2_var'].median())
#%%
#price_p3_var
price_p3_var = df4['price_p3_var'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['price_p3_var'] = df4['price_p3_var'].fillna(df4['price_p3_var'].median())
#%%
#price_p1_fix
price_p1_fix = df4['price_p1_fix'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['price_p1_fix'] = df4['price_p1_fix'].fillna(df4['price_p1_fix'].median())
#%%
# Delete the negative number 
df4.drop(df4[df4['price_p1_fix'] < 0].index, axis=0, inplace=True)
#%%
#price_p2_fix
price_p2_fix = df4['price_p2_fix'].value_counts(dropna=False)
#%%
#Replace the missing values with its median
df4['price_p2_fix'] = df4['price_p2_fix'].fillna(df4['price_p2_fix'].median())

#%%
#price_p3_fix
price_p3_fix = df4['price_p3_fix'].value_counts(dropna=False)

#%%
#Replace the missing values with its median
df4['price_p3_fix'] = df4['price_p3_fix'].fillna(df4['price_p3_fix'].median())
#%%
#churn
churn = df4['churn'].value_counts(dropna=False)
#%%

# Check whether it has eliminated all the missing values
missing_values_df4 = df4.isna().sum()
missing_values_df4

#%%
# Encoding categorical variables

# Select only categorical features
features_categoriche = df4.select_dtypes(include="object")

#%%
# channel_sales
channel_sales_le = LabelEncoder()
df4['channel_sales'] = channel_sales_le.fit_transform(df4['channel_sales'])
channel_sales_le.classes_
df4['channel_sales'].value_counts(dropna=False)

#%%

# date_modif_prod

date_modif_prod = LabelEncoder()
df4['date_modif_prod'] = date_modif_prod.fit_transform(df4['date_modif_prod'])
date_modif_prod.classes_
df4['date_modif_prod'].value_counts(dropna=False)

#%%
# date_renewal
date_renewal_le = LabelEncoder()
df4['date_renewal'] = date_renewal_le.fit_transform(df4['date_renewal'])
date_renewal_le.classes_
df4['date_renewal'].value_counts(dropna=False)
#%%
#has_gas

df4['has_gas'].value_counts()
m = {
    "f": 0,
    "t": 1,
}
df4['has_gas'] = df4['has_gas'].map(m)
#%%
#origin_up
origin_up_le = LabelEncoder()
df4['origin_up'] = origin_up_le.fit_transform(df4['origin_up'])
origin_up_le.classes_
df4['origin_up'].value_counts(dropna=False)
#%%

# Check to see if there are no more categorical variables
df4.info()
#%%
df4_dict = df4.to_dict('records')
df4_list_bal = list()
count=0
for row in df4_dict:
    if row['churn'] == 0:
        count += 1
        if count<=17730:
            df4_list_bal.append(row)
    else:
        df4_list_bal.append(row)
        
df4_new = pd.DataFrame(df4_list_bal)

#%%
# Extraction of the y
y = df4_new.churn
df4_new = df4_new.drop('churn', axis=1)

#%%
# Check if y has missing values
y.isna().sum()  # no
#%%

# Check if I have an unbalanced dataset
y.value_counts()
z = y.value_counts()
xdata3 = ['0', '1']

fig2 = plt.figure(figsize=(10, 7))

plt.bar(xdata3, z)
plt.xticks(np.linspace(0, 12, 13, endpoint=True))
plt.show()

# Is somewhat unbalanced in favor of the 0(clienti che hanno abbandonato), penalizing the 1

#%%
# **Statistical Analysis**

# Correlation of problematic variables with churn
corr = df4_new.corr()
#%%
# Correlation between features and churn
df4_new['channel_sales'].corr(y)
df4_new['cons_12m'].corr(y)
df4_new['cons_gas_12m'].corr(y)
df4_new['cons_last_month'].corr(y)
df4_new['date_modif_prod'].corr(y)
df4_new['date_renewal'].corr(y)
df4_new['forecast_cons_12m'].corr(y)
df4_new['forecast_cons_year'].corr(y)
df4_new['forecast_discount_energy'].corr(y)
df4_new['forecast_meter_rent_12m'].corr(y)
df4_new['forecast_price_energy_p1'].corr(y)
df4_new['forecast_price_energy_p2'].corr(y)
df4_new['forecast_price_pow_p1'].corr(y)
df4_new['has_gas'].corr(y)
df4_new['imp_cons'].corr(y)
df4_new['margin_gross_pow_ele'].corr(y)
df4_new['margin_net_pow_ele'].corr(y)
df4_new['nb_prod_act'].corr(y)
df4_new['net_margin'].corr(y)
df4_new['num_years_antig'].corr(y)
df4_new['origin_up'].corr(y)
df4_new['pow_max'].corr(y)
df4_new['price_p1_var'].corr(y)
df4_new['price_p2_var'].corr(y)
df4_new['price_p3_var'].corr(y)
df4_new['price_p1_fix'].corr(y)
df4_new['price_p2_fix'].corr(y)
df4_new['price_p3_fix'].corr(y)
#%%
# Correlation between features of the dataset
rounded_corr_matrix = df4_new.corr().round(2)    
#%%
# Heatmap on all variables
heatmap = sns.heatmap(rounded_corr_matrix, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
#%%

# We choose few features, the central ones in the overall heatmap, the ones that are most related
features = ["forecast_cons_12m", "forecast_cons_year",
            "forecast_discount_energy", "forecast_meter_rent_12m", "forecast_price_energy_p1",
            "forecast_price_energy_p2", "forecast_price_pow_p1", "net_margin",
            "num_years_antig","origin_up", "pow_max", "price_p1_var", "price_p2_var", "price_p3_var", "price_p1_fix", "price_p2_fix", "price_p3_fix"]

subset = rounded_corr_matrix[features].loc[features]
heatmap1 = sns.heatmap(subset, annot=True)
#%%
# Delete forecast_discount_energy,num_years_antig and origin_up
features2 = ["forecast_cons_12m", "forecast_cons_year",
             "forecast_meter_rent_12m", "forecast_price_energy_p1",
             "forecast_price_energy_p2", "forecast_price_pow_p1", "net_margin",
             "pow_max", "price_p1_var", "price_p2_var", "price_p3_var", "price_p1_fix", "price_p2_fix", "price_p3_fix"]


subset2 = rounded_corr_matrix[features2].loc[features2]
heatmap2 = sns.heatmap(subset2, annot=True)
#%%

# ACP
# Standardized all values
sc = StandardScaler()
df_scaled = sc.fit_transform(df4_new)
#%%
pca = PCA()
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))
df_pca
#%%
import matplotlib.pyplot as plt
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance");

#%%
# Pick 6 components
pca = PCA(n_components=6)

df6 = pca.fit_transform(df_scaled)

explained_variance = pca.explained_variance_ratio_
#%%
# Choice of components based on 95% variance
pca = PCA(.95)

df7 = pca.fit_transform(df_scaled)

explained_variance = pca.explained_variance_ratio_

pca.n_components_  
#%%
# Try with 90% of variance
pca = PCA(.90)

df8 = pca.fit_transform(df_scaled)

explained_variance = pca.explained_variance_ratio_

pca.n_components_
#%%
# **Features Selection**
# Define x
x = df4_new

#%%
#ANOVA UNIVARIATE Feature Selection
# Split dataset to select feature and evaluate the classifier
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)

#%%
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=4)
selector.fit(x_train, y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
#%%
import matplotlib.pyplot as plt

x_indices = np.arange(x.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(x_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()
#%%
# Delete worst variables by score 
df4_new.drop(['forecast_cons_12m', 'forecast_cons_year', 'forecast_price_energy_p1',
        'imp_cons', 'pow_max'], axis=1, inplace=True)
#%%
# Define x
x = df4_new
#%%
#MUTUAL INFORMATION feature selecion
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=0)

from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(x_train, y_train)
mutual_info
#%%
mutual_info = pd.Series(mutual_info)
mutual_info.index = x_train.columns
mutual_info.sort_values(ascending=False)
#%%
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))

#%%
from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=5)
sel_five_cols.fit(x_train, y_train)
x_train.columns[sel_five_cols.get_support()]
#%%
# Delete variables with mutual information less than 0.001
df4_new.drop(['forecast_discount_energy', 'has_gas', 'nb_prod_act',
        'price_p3_fix', 'price_p2_fix'], axis=1, inplace=True)
#%%
# **Machine Learning Models**

#RANDOM FOREST
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
#%%
rfc = RandomForestClassifier(n_estimators=2, 
                             max_depth=2,
                             random_state=0)
#%%
# Hyperparameters 
p_grid = {"max_depth": [2, 3, 4, 5, 10, 15, 20, 25, 30]}

# Objective of validation stage is to tune / optimize the hyperparameter
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# We perform the search of the best hyperparameters within the validation set
rfc = GridSearchCV(estimator=rfc, param_grid=p_grid, cv=inner_cv, verbose=1)
#%%
# Fit RandomForestClassifier
rfc.fit(x_train, y_train)
# Predict the test set labels
y_pred = rfc.predict(x_test)
#%%
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('confusion matrix ')

print(classification_report(y_test,y_pred))
#%%
# ROC Curve
ypredpropa = rfc.predict_proba(x_test)    

n_class = 2   
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, ypredpropa[:, i], pos_label=i)

# Plotting ROC
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
plt.title("Multiclass ROC curve rfc", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")

#%%
# AUC
#roc_auc_score(y_test, rfc.predict_proba(x_test), multi_class='ovr')

from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# DECISION TREE
# Create training and testing samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

dt = DecisionTreeClassifier(random_state=0)

# Hyperparameters 
p_grid = {"max_depth": [2, 3, 4, 5, 10, 15, 20, 25, 30]}

# Objective of validation stage is to tune / optimize the hyperparameter
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# We perform the search of the best hyperparameters within the validation set
clf = GridSearchCV(estimator=dt, param_grid=p_grid, cv=inner_cv, verbose=1)
#%%
# Train the model 
clf.fit(x_train, y_train)
#%%
print('Best max_depth:', clf.best_estimator_.get_params()['max_depth'])
#%%
# Accuracy of train
clf_score_train=clf.score(x_train , y_train)
clf_score_train
#%%
# Prediction
y_pred = clf.predict(x_test)

# Checking performance of our model with classification report
print(classification_report(y_test, y_pred))
#%%
# Plot our Confusion Matrix
plot_confusion_matrix(clf, x_test, y_test, cmap='Blues')
#%%
# ROC Curve
ypredpropa = clf.predict_proba(x_test)    

n_class = 2    
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, ypredpropa[:, i], pos_label=i)

# Plotting ROC
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
plt.title("Multiclass ROC curve clf", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")

plt.savefig("Multiclass ROC_pipe2", dpi=300)   

#%%
from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


#%%
#Naive-Bayes
# Create training and testing samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#%%
from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(x_train, y_train)

#%%
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(x_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

# Checking performance of our model with classification report
print(classification_report(y_test, y_pred))
#%%
labels = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();
#%%
# ROC Curve
ypredpropa = clf.predict_proba(x_test)    

n_class = 2    
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, ypredpropa[:, i], pos_label=i)

# Plotting ROC
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
plt.title("Multiclass ROC curve clf", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")

plt.savefig("Multiclass ROC_pipe2", dpi=300)   
#%%
#AUC
# AUC
#roc_auc_score(y_test, rfc.predict_proba(x_test), multi_class='ovr')

from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
#%%
# Splitting our dataset into training and test set (hold-out)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
[x_train.shape, y_train.shape, x_test.shape, y_test.shape]

#%%
# KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Hyperparameters
p_grid = {"n_neighbors": [2, 3, 5, 4, 6, 7, 8, 9, 10],}

# Objective of validation stage is to tune / optimize the hyperparameter
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# We perform the search of the best hyperparameters within the validation set
knnc = GridSearchCV(knn, p_grid, n_jobs=-1, cv=inner_cv, verbose=10, scoring="accuracy")

# Train the model
knnc.fit(x_train, y_train)

print('n_neighbors:', knnc.best_estimator_.get_params()['n_neighbors'])

# Accuracy of train
knnc.score(x_train, y_train)

# Prediction
ypred = knnc.predict(x_test)

# Checking performance of our model with classification report
print(classification_report(y_test, ypred))

# Plot our Confusion matrix
plot_confusion_matrix(knnc, x_test, y_test, cmap='Blues')

# ROC Curve
ypredpropa = knnc.predict_proba(x_test)    

n_class = 2    
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, ypredpropa[:, i], pos_label=i)

# Plotting ROC
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
#plt.plot(fpr[2], tpr[2], linestyle="--", color="blue", label="Class 2 vs Rest")
plt.title("Multiclass ROC curve knnc", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")
#%%
# AUC
roc_auc_score(y_test, knnc.predict_proba(x_test)[:, 1], multi_class='ovr')
#%%
# AUC
#roc_auc_score(y_test, rfc.predict_proba(x_test), multi_class='ovr')

from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()



































