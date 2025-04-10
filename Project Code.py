#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Step 1: Data Exploration
import pandas as pd


# Loading and Inspect of Datasets
production_data = pd.read_csv(r'E:\wine\New Data Wine Production.csv')
consumption_data = pd.read_csv(r'E:\wine\New Data Wine Consumption.csv')
print("Production Data Head -")
print(production_data.head(), "\n")
print("Consumption Data Head -")
print(consumption_data.head(), "\n")


# Checking for Missing Values and Duplicate Rows:
print("Missing Values in Production Data -")
print(production_data.isnull().sum(), "\n")
print("Missing Values in Consumption Data -")
print(consumption_data.isnull().sum(), "\n")
print("Duplicate Rows in Production Data -")
print(production_data.duplicated().sum(), "\n")
print("Duplicate Rows in Consumption Data -")
print(consumption_data.duplicated().sum(), "\n")


# Getting Summary Stats:
print("Summary Statistics for Production Data -")
print(production_data.describe(), "\n")
print("Summary Statistics for Consumption Data -")
print(consumption_data.describe(), "\n")


# Checking Unique Values in Key Columns:
print("Unique Areas in Production Data -")
print(production_data['Area'].unique(), "\n")
print("Unique Areas in Consumption Data -")
print(consumption_data['Area'].unique(), "\n")
print("Years Covered in Production Data -")
print(production_data['Year'].unique(), "\n")
print("Years Covered in Consumption Data -")
print(consumption_data['Year'].unique(), "\n")


# In[ ]:


# Step 2: Data Cleaning
print("Handling Missing Values in Production Data")


# Filling Missing Values with the Mean of the Column & Verifying Missing Values
production_data['Value'] = production_data['Value'].fillna(production_data['Value'].mean())
print("Missing Values in Production Data after cleaning:")
print(production_data.isnull().sum(), "\n")


# Checking Outliers Values in 'Value' Column
print("Checking for outliers in Production Data")
print(production_data.describe())  


# Ensuring both datasets have consistent year ranges and Merging the datasets for analysis
print("Ensuring year range consistency")
print("Years in Production Data:", production_data['Year'].unique())
print("Years in Consumption Data:", consumption_data['Year'].unique())
print("Merging Production and Consumption Data")


# Checking and Save the merged data
merged_data = pd.merge(production_data, 
                       consumption_data, 
                       on=['Area', 'Year', 'Item'], 
                       suffixes=('_Production', '_Consumption'))
print("Merged Data - Head:")
print(merged_data.head(), "\n")
merged_data.to_csv(r'E:\wine\Merged_Wine_Data.csv', index=False)
print("Cleaned and Merged Data saved to 'Merged_Wine_Data.csv'")


# In[ ]:


# Step 3: Data Analysis and Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Key Metrics Analysis: Total production and consumption by region
region_summary = merged_data.groupby('Area')[['Value_Production', 'Value_Consumption']].sum().reset_index()
print("Total Production and Consumption by Region:")
print(region_summary, "\n")


# Trend Analysis: Total production and consumption by year
year_summary = merged_data.groupby('Year')[['Value_Production', 'Value_Consumption']].sum().reset_index()
print("Total Production and Consumption by Year:")
print(year_summary, "\n")


# Correlation Analysis: between production and consumption
correlation = merged_data['Value_Production'].corr(merged_data['Value_Consumption'])
print(f"Correlation between Production and Consumption: {correlation:.2f}\n")


# Visualization 1
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
sns.lineplot(data=year_summary, x='Year', y='Value_Production', label='Production', marker='o', ax=ax1, color='tab:blue')
ax1.set_xlabel('Year')
ax1.set_ylabel('Production Value', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('Wine Production and Consumption Trends (2018-2022)')
ax2 = ax1.twinx()  
sns.lineplot(data=year_summary, x='Year', y='Value_Consumption', label='Consumption', marker='o', ax=ax2, color='tab:orange')
ax2.set_ylabel('Consumption Value (kg/capita)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid()
plt.show()


# Visualization 2
bar_width = 0.35
index = np.arange(len(region_summary))  
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(index, region_summary['Value_Production'], bar_width, label='Production', color='tab:blue')
ax1.set_xlabel('Region')
ax1.set_ylabel('Production Value', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(index)
ax1.set_xticklabels(region_summary['Area'], rotation=45, ha='right')
ax2 = ax1.twinx()
ax2.bar(index + bar_width, region_summary['Value_Consumption'], bar_width, label='Consumption', color='tab:orange')
ax2.set_ylabel('Consumption Value (kg/capita)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
fig.tight_layout()
plt.title('Total Wine Production and Consumption by Region')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.85))
plt.grid(axis='y')
plt.show()


# Visualization 3
plt.figure(figsize=(8, 6))
sns.scatterplot(data=merged_data, x='Value_Production', y='Value_Consumption', hue='Area', s=100, alpha=0.7)
plt.title('Correlation Between Production and Consumption')
plt.xlabel('Production Value')
plt.ylabel('Consumption Value')
plt.grid()
plt.legend(title='Region')
plt.tight_layout()
plt.show()



# In[7]:


# Calculate and Visualize Surplus/Deficit (Production - Consumption)
region_summary['Surplus/Deficit'] = region_summary['Value_Production'] - region_summary['Value_Consumption']
plt.figure(figsize=(12, 6))
sns.barplot(data=region_summary, x='Area', y='Surplus/Deficit')
plt.xlabel('Region')
plt.ylabel('Surplus/Deficit (Production - Consumption)')
plt.xticks(rotation=45, ha='right')
plt.title('Wine Surplus/Deficit by Region')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

