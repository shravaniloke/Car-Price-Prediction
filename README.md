# Car-Price-Prediction
# 🚗 Car Price Prediction using Machine Learning

This project uses a dataset of used cars to build a machine learning model that predicts car prices based on features like year, fuel type, seller type, transmission, and more.


## 📌 Project Overview

With the growing used car market, pricing a car fairly is crucial for both buyers and sellers. This project aims to:
- Explore the dataset with visualizations
- Preprocess data (handle categorical features, missing values)
- Train a regression model to predict selling price
- Evaluate model performance with proper metrics


## 📁 Files Included

 **car_price_prediction.ipynb**  Main Google Colab notebook with complete code, visuals, and model |
 **Car details v3.csv**          The dataset containing car listings and their details |


## 📊 Dataset Summary

- Number of records: `~8,000+`
- Key columns: `Year`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`, `Selling_Price`


## 🧠 Machine Learning Model

- **Model Used**: Linear Regression  
- **Target Variable**: `Selling_Price`
- **Features**: Numeric + Categorical (encoded)
- **Evaluation Metrics**:
  - R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)


## 📉 Results

> ✅ The final model achieved an R² score of **0.4857** on the test data, indicating moderate predictive power for car price estimation.


## 🔧 Tech Stack

- **Language**: Python 3
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
- **Platform**: Google Colab


## 📷 Sample Visualizations

- Distribution of Selling Prices
- Fuel Type vs Price
- Heatmap of Correlations


## 🚀 How to Run

1. Clone this repository or download the files.
2. Open `car_price_prediction.ipynb` in Google Colab or Jupyter Notebook.
3. Make sure `Car details v3.csv` is in the same directory.
4. Run all cells to train the model and view results.


## 🤝 Credits

Dataset: [[Kaggle - Car Details Dataset](https://www.kaggle.com/datasets/)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download)

Project by: Shravani Loke


