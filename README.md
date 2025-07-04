# Music Genre Prediction

## Abstract
This project focuses on predicting music genres using audio features from Spotify tracks. The analysis involves exploring various features like tempo, energy, loudness, acousticness, danceability, and more. The project utilizes data visualization techniques (histograms, box plots) to understand data characteristics, identify missing values, and assess the usefulness of features. H2O.ai AutoML is employed for model building and interpretation, including assessing relationship significance, multicollinearity, and using SHAP analysis for feature importance. The project addresses a multi-class classification problem for genres like pop, rock, jazz, hip-hop, electronic, classical, and country, aiming for a comprehensive understanding of model interpretability using SHAP.

## Dataset
The dataset contains 50,000 observations with 18 columns, including:
- `instance_id`
- `artist_name`
- `track_name`
- `popularity`
- `acousticness`
- `danceability`
- `duration_ms`
- `energy`
- `instrumentalness`
- `key`
- `liveness`
- `loudness`
- `mode`
- `speechiness`
- `tempo`
- `obtained_date`
- `valence`
- `music_genre` (Target Variable)

The genres included are: 'Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Rock', 'Classical', 'Hip-Hop'.

## Overview and Flow
The project follows a standard machine learning workflow including data loading, cleaning, exploratory data analysis, feature engineering, model training, evaluation, and interpretation.

## Data Cleaning and Preprocessing
- Handled missing values, particularly in the 'tempo' column using KNN imputation.
- Removed high entropy features like `instance_id`, `track_name`, and `obtained_date`.
- Addressed outliers in numerical features using RobustScaler.
- Applied Label Encoding to categorical variables (`key`, `mode`, `artist_name`, `music_genre`).
- Assessed multicollinearity using Variance Inflation Factor (VIF) and removed the 'energy' column due to high correlation with 'loudness'.

## Exploratory Data Analysis
Visualizations were used to understand the distribution of numerical and categorical features.
- The target variable 'music_genre' is balanced across 10 genres.
- The 'key' and 'mode' distributions were examined.
- Distributions of numerical features were visualized using histograms and box plots.

## Feature Selection
Various methods were used to identify important features:
- **Correlation Matrix:** Examined correlations between features and the target variable, and among predictor variables.
- **Recursive Feature Elimination (RFE):** Identified 'popularity', 'acousticness', 'danceability', 'loudness', and 'speechiness' as important.
- **Permutation Importance:** Highlighted popularity, danceability, instrumentalness, acousticness, and loudness as important.
- **OLS Regression:** Assessed the statistical significance of features.
- **XGBoost Feature Importance:** Identified popularity, loudness, instrumentalness, speechiness, and mode as the most important features.
- **SHAP Analysis:** Provided insights into the contribution of each feature to the model's predictions, consistently showing popularity, instrumentalness, danceability, and loudness as influential.

## ML Models
Several models were trained and evaluated:
- **Tree-Based Models (Random Forest, ADA, XGBoost):** XGBoost performed best, especially after hyperparameter tuning and including the encoded 'artist_name'.
- **Linear Model (Multinomial Logistic Regression):** Provided a baseline and interpretable coefficients. Ridge regularization was applied but showed only a marginal improvement in RMSE.
- **AutoML (H2O.ai):** A Stacked Ensemble model emerged as the best performing model on the leaderboard, outperforming individual base models (GBM, XGBoost, GLM).

## Model Interpretability
- **Confusion Matrices:** Generated for XGBoost and Logistic Regression to visualize classification performance across genres.
- **SHAP Analysis:** Used to interpret the predictions of XGBoost, Logistic Regression, and the H2O AutoML best model, providing insights into individual feature contributions.

## Inferences
- The inclusion of encoded 'artist_name' improved the performance of tree-based models.
- Hyperparameter tuning significantly narrowed the performance gap between models trained with and without outliers.
- Popularity, instrumentalness, danceability, and loudness consistently appeared as important features across different feature selection and model interpretation methods.
- The H2O AutoML Stacked Ensemble model achieved the best overall performance.

## Conclusion
The project successfully built and interpreted models for music genre prediction. The analysis highlighted the importance of various audio features and the artist's influence. AutoML proved effective in finding a high-performing model. SHAP analysis was crucial for understanding model decisions and confirming the relevance of key features.

## License
MIT License

Copyright (c) 2024 Malini Janaki Sankaran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## References
1. https://towardsdatascience.com/6-predictive-models-models-every-beginner-data-scientist-should-master-7a37ec8da76d
2. https://www.software.com/src/explore-the-data-behind-your-most-productive-music-for-coding
3. https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html
4. https://towardsdatascience.com/a-deep-dive-into-h2os-automl-4b1fe51d3f3e
5. https://www.analyticsvidhya.com/blog/2021/02/unboxing-h2o-automl-models/
6. https://towardsdatascience.com/automated-machine-learning-with-h2o-258a2f3a203f
7. https://towardsdatascience.com/automated-machine-learning-with-h2o-258a2f3a203f
8. https://www.youtube.com/watch?v=MQ6fFDwjuco
9. https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/
10. https://www.analyticsvidhya.com/blog/2021/02/unboxing-h2o-automl-models/
