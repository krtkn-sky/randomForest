# Random Forest Regression for Performance Rating Prediction

## Overview

This code uses Random Forest Regression to predict performance ratings based on years of experience. It also includes 5-fold cross-validation to assess the model's performance.

## Dependencies

Before running the code, ensure you have the following libraries installed:
- numpy
- matplotlib
- pandas
- scikit-learn (sklearn)

You can install these dependencies using pip if you haven't already:

```bash
pip install numpy matplotlib pandas scikit-learn
```

## Instructions

1. **Dataset**: Ensure you have a CSV dataset file ('Ratings.csv') with columns for 'Years of Experience' (independent variable) and 'Performance Rating' (dependent variable).

2. **Training and Prediction**: The code reads the dataset, trains a Random Forest Regression model, and predicts the performance rating for a specific 'Years of Experience' value (e.g., 6.5).

3. **Visualization**: The code generates a visual plot of the regression model's predictions against the actual data.

4. **Cross-Validation**: The code performs 5-fold cross-validation to evaluate the model's performance using RMSE (Root Mean Squared Error) scores.
