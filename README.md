
# Flight Price Prediction - EDA and Feature Engineering

## Project Overview
This project focuses on **Exploratory Data Analysis (EDA)** and **Feature Engineering** for flight price prediction. 
The goal is to clean, transform, and analyze flight data to build a reliable model for predicting flight ticket prices.

## Dataset
- The project uses a flight pricing dataset, which includes features such as airline, date of journey, source, destination, duration, and price.

## Steps Included
1. **Data Cleaning:** Handling missing values, correcting data types.
2. **Feature Engineering:** Extracting new features, encoding categorical variables.
3. **Exploratory Data Analysis:** Visualizing patterns and trends.
4. **Model Preparation:** Preparing the dataset for machine learning models.

## Sample Code
```python
#importing basics libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_excel('/flight_price.xlsx')

df.head()
df.tail()
```

## How to Run
1. Clone the repository:  
```

2. Install dependencies:  
```
pip install -r requirements.txt
```
3. Open the Jupyter notebook:  
```
jupyter notebook flight_price_prediction.ipynb
```

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
