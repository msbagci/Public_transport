# Public_transport
Public transport usage rates in Ukraine by year

### 🔎 Data Source Attribution

The dataset used for this analysis is publicly available on the Kaggle platform. This is an essential step for reproducibility and adhering to open-source data ethics.

* **Dataset Name:** [UA Passengers 1995-2020]
* **Original Source URL:** [https://www.kaggle.com/datasets/kseniakryvogubchenko/ua-passengers-1995-2020]

🇺🇦 UKRAINIAN PUBLIC TRANSPORTATION TREND ANALYSIS (1995-2020)

This project examines the passenger volume trends across various public transport modes in Ukraine (railway, bus, air, subway, etc.) between 1995 and 2020. The primary goal is to use statistical methods to analyze the long-term trend, particularly focusing on a major disruption observed in the mid-2010s.

🛠️ Technologies Used
Python: Main programming language.

Pandas: Data manipulation and transformation.

Seaborn/Matplotlib: Data visualization.

scikit-learn: Linear Regression modeling.

Key Analysis Steps
1. Data Preparation
The raw data (transportation types as separate columns) was converted from wide format to long format using pd.melt. This transformation was essential for running time series analysis and creating multi-line plots using the hue parameter.

2. Exploratory Data Analysis (EDA)
Visualization of yearly passenger volume revealed a stable trend until 2014, followed by a sharp, sustained decline. This indicated a severe structural break in the time series.

Initial attempts to model the overall trend using standard Linear Regression failed, yielding a significantly negative R-squared ($\approx -0.11$). This result is expected when a model trained on stable data attempts to predict a period of sudden, permanent change.

Solution: Piecewise Linear RegressionTo address the structural break, the following dummy variables were introduced:

Indicator_2014: A dummy variable used to capture the intercept shift (sudden change in the level) after 2014.

Trend_After_2014: A dummy variable used to model a change in the slope (trend) starting from 2014.

Simple Linear Model$R^2 \approx -0.11$

Piecewise Regression$R^2 \approx 0.80$

Conclusion: The high $R^2$ score of the Piecewise Regression model demonstrates that the inclusion of specific statistical indicators successfully explained the complex, disrupted passenger volume trend.

🚀 How to Run

Clone the repository and place the ua_passengers_1995-2020.csv file in the main folder.

Run the Jupyter Notebook file to reproduce the analysis and all visualizations.

This project is licensed under the MIT License. Please see the (LICENCE)[LICENCE] file for more details.

