import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import time

def get_model_and_param_grid(model_type):
    if model_type == "LinearRegression":
        model = LinearRegression()
        apply_scaling = True
        param_grid = {}
    elif model_type == "Ridge":
        model = Ridge()
        apply_scaling = True
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_type == "Lasso":
        model = Lasso()
        apply_scaling = True
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_type == "DecisionTree":
        model = DecisionTreeRegressor()
        apply_scaling = False
        param_grid = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'max_depth': [None, 10, 20, 30, 40, 50]
        }
    elif model_type == "KNeighbors":
        model = KNeighborsRegressor()
        apply_scaling = True
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
    elif model_type == "SVR":
        model = SVR()
        apply_scaling = True
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
    elif model_type == "AdaBoost":
        model = AdaBoostRegressor()
        apply_scaling = False
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        }
    elif model_type == "XGBoost":
        model = XGBRegressor()
        apply_scaling = False
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        raise ValueError("Invalid model type provided.")
    
    return model, param_grid, apply_scaling

def scale_features(X, apply_scaling):
    if apply_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def perform_ensemble_grid_search(X_train, y_train):
    model1, param_grid1, _ = get_model_and_param_grid('Ridge')
    model2, param_grid2, _ = get_model_and_param_grid('DecisionTree')
    model3, param_grid3, _ = get_model_and_param_grid('XGBoost')
    
    model1 = perform_grid_search(model1, param_grid1, X_train, y_train)
    model2 = perform_grid_search(model2, param_grid2, X_train, y_train)
    model3 = perform_grid_search(model3, param_grid3, X_train, y_train)
    
    ensemble_model = StackingRegressor(estimators=[('ridge', model1), ('dt', model2), ('xgb', model3)])
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def evaluate_metrics(y_true, y_pred, n_features):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'adjusted_r2': adjusted_r2
    }
    return metrics

def plot_train_vs_test_metrics(train_metrics, test_metrics, model_name):
    metrics = ['mse', 'rmse', 'mae', 'r2']
    
    for metric in metrics:
        train_value = train_metrics[metric]
        test_value = test_metrics[metric]

        fig, ax = plt.subplots()
        bars1 = ax.bar('Train', train_value, width=0.35, label='Train')
        bars2 = ax.bar('Test', test_value, width=0.35, label='Test', color='orange')

        ax.set_ylabel(metric.upper())
        ax.set_title(f'{model_name} - Train vs Test {metric.upper()}')
        ax.legend()

        # Adding bar labels
        for bars in [bars1, bars2]:
            ax.bar_label(bars, padding=3)

        plt.show()

def evaluate_regression_model(df, target_var, feature_list, model_type, run_ensemble_mode=False):
    X = df[feature_list]
    y = df[target_var]
    start_time = time.time()
    
    try:
        if not run_ensemble_mode:
            model, param_grid, apply_scaling = get_model_and_param_grid(model_type)
            X_transformed = scale_features(X, apply_scaling)  # Only scaling the features now
            X_train, X_test, y_train, y_test = split_data(X_transformed, y)
            optimized_clf = perform_grid_search(model, param_grid, X_train, y_train)

        if run_ensemble_mode:
            X_train, X_test, y_train, y_test = split_data(X, y)  # No transformation applied here
            optimized_clf = perform_ensemble_grid_search(X_train, y_train)

        train_preds = optimized_clf.predict(X_train)
        test_preds = optimized_clf.predict(X_test)

        train_metrics = evaluate_metrics(y_train, train_preds, X_train.shape[1])
        test_metrics = evaluate_metrics(y_test, test_preds, X_train.shape[1])

        print(f"Model: {model_type if not run_ensemble_mode else 'Ensemble'}")
        print("The RMSE on train data is ", train_metrics['rmse'])
        print("The RMSE on test data is ", test_metrics['rmse'])
        print("The MSE on train data is ", train_metrics['mse'])
        print("The MSE on test data is ", test_metrics['mse'])
        print("The MAE on train data is ", train_metrics['mae'])
        print("The MAE on test data is ", test_metrics['mae'])
        print("The R2 on train data is ", train_metrics['r2'])
        print("The R2 on test data is ", test_metrics['r2'])
        print("The Adjusted R2 on train data is ", train_metrics['adjusted_r2'])
        print("The Adjusted R2 on test data is ", test_metrics['adjusted_r2'])

        # Plot train vs test metrics
        plot_train_vs_test_metrics(train_metrics, test_metrics, model_type if not run_ensemble_mode else 'Ensemble')

        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return {
            'model_type': model_type if not run_ensemble_mode else 'Ensemble',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'test_preds': test_preds,
            'y_test': y_test
        }
    except ValueError as e:
        print(e)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return None


def evaluate_regression_models(models, df, target_var, feature_list):
    
    results = []

    run_ensemble_mode = False

    # Run for all models
    for model in models:
        print('model type == ' + model + '  ..STARRING - evaluate_regression_model func')
        result = evaluate_regression_model(df, target_var, feature_list, model, run_ensemble_mode)
        if result is not None:
            results.append(result)
        print('model type == ' + model + '  ..END - evaluate_regression_model func')
    
    # Run for Ensemble model if applicable
    run_ensemble_mode = True
    if run_ensemble_mode:
        print('model type == Ensemble  ..STARRING - evaluate_regression_model func')
        result = evaluate_regression_model(df, target_var, feature_list, 'Ensemble', run_ensemble_mode)
        if result is not None:
            results.append(result)
        print('model type == Ensemble  ..END - evaluate_regression_model func')

    # Convert results to a DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'model_type': [res['model_type'] for res in results],
        'test_rmse': [res['test_metrics']['rmse'] for res in results],
        'test_mse': [res['test_metrics']['mse'] for res in results],
        'test_mae': [res['test_metrics']['mae'] for res in results],
        'test_r2': [res['test_metrics']['r2'] for res in results],
        'test_adjusted_r2': [res['test_metrics']['adjusted_r2'] for res in results]
    })

    # Plot the metrics comparison for all models
    metrics = ['rmse', 'mse', 'mae', 'r2', 'adjusted_r2']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y=f'test_{metric}', data=metrics_df)
        plt.title(f'Test {metric.upper()} Comparison')
        plt.ylabel(f'Test {metric.upper()}')
        plt.xlabel('Model Type')
        plt.xticks(rotation=45)
        plt.show()

# Example usage
# Note: You'll need to load your dataset before using these functions

# target_var = 'Initial Cost'
# feature_list = ['Job','Lot','Job Type','Proposed Zoning Sqft','Enlargement SQ Footage','year']
# models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree',  'KNeighbors', 'SVR', 'AdaBoost', 'XGBoost'] 
# evaluate_regression_models(models, ny_df, target_var, feature_list)
