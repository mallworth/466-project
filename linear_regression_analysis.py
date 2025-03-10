import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def safe_read_csv(file_path):
    """Enhanced CSV reading with handling for dataset title in first line"""
    try:
        # Check the first few lines
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()[:5]]
        
        # Check if the first line is a dataset title (not a header)
        # This is detected if the second line contains many commas (column names)
        if len(lines) >= 2:
            if lines[0].count(',') <= 1 and lines[1].count(',') > 1:
                # Skip the first line (dataset title) and read from the second line
                return pd.read_csv(file_path, skiprows=1)
            else:
                # Standard CSV reading
                return pd.read_csv(file_path)
        else:
            # Not enough lines to determine, use standard reading
            return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def linear_regression_analysis():
    """Apply linear regression to predict streaming counts or popularity"""
    data_dir = './data'
    processed_dir = './data/processed'
    output_dir = './results/regression'
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for datasets with streams or popularity information
    datasets = {}
    target_columns = ['streams', 'popularity']
    
    # Check both data and processed directories
    search_dirs = [data_dir]
    if os.path.exists(processed_dir):
        search_dirs.append(processed_dir)
    
    for dir_path in search_dirs:
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                file_path = os.path.join(dir_path, file)
                name = file.replace('_processed.csv', '').replace('.csv', '')
                
                # Use safe_read_csv to handle the dataset title in first line
                df = safe_read_csv(file_path)
                
                if df is None:
                    continue
                
                # Check if dataset has a valid target column
                for col in target_columns:
                    if col in df.columns:
                        try:
                            # Convert to numeric if needed
                            if not pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Drop rows with missing target values
                            df = df.dropna(subset=[col])
                            
                            if len(df) > 100:  # Only use datasets with enough samples
                                datasets[(name, col)] = df
                                print(f"Found {name} dataset with {len(df)} samples for {col} prediction")
                        except Exception as e:
                            print(f"Error processing {col} in {file}: {e}")
    
    if not datasets:
        print("No suitable datasets found with streams or popularity information")
        return
    
    # Use the dataset with the most samples
    dataset_key = max(datasets.keys(), key=lambda k: len(datasets[k]))
    dataset_name, target_col = dataset_key
    df = datasets[dataset_key]
    
    print(f"Selected {dataset_name} with {len(df)} samples for predicting {target_col}")
    
    # Select features
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo', 'duration_ms']
    
    # Keep only features that exist in the dataframe
    features = [f for f in audio_features if f in df.columns]
    
    if len(features) < 3:
        print("Not enough audio features for meaningful regression")
        return
    
    print(f"Using {len(features)} features: {features}")
    
    # Identify and handle categorical features
    numeric_features = []
    categorical_features = []
    
    for feature in features:
        # Check if feature is categorical
        if pd.api.types.is_object_dtype(df[feature]) or (feature in ['key', 'mode'] and not pd.api.types.is_numeric_dtype(df[feature])):
            categorical_features.append(feature)
        else:
            try:
                # Try to convert to numeric just to be sure
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                numeric_features.append(feature)
            except:
                # If conversion fails, treat as categorical
                categorical_features.append(feature)
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Check target distribution
    plt.figure(figsize=(10, 6))
    if target_col == 'streams' and df[target_col].max() > 1000000:
        # Use log scale for streams
        df['log_target'] = np.log10(df[target_col] + 1)  # Add 1 to handle zeros
        sns.histplot(df['log_target'], kde=True)
        plt.title(f'Distribution of log10({target_col})')
        plt.xlabel(f'log10({target_col})')
        regression_target = 'log_target'
    else:
        sns.histplot(df[target_col], kde=True)
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        regression_target = target_col
    
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    
    # Prepare data - handle categorical features properly
    X = df[features].copy()
    y = df[regression_target]
    
    # Fill missing values
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown')
    
    # Create preprocessor for both numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train multiple regression models
    models = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=1.0))
        ]),
        'Lasso Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(alpha=0.1))
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'cv_scores': cv_scores,
                'y_pred': y_pred
            }
            
            print(f"\n{name} results:")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Mean CV R²: {np.mean(cv_scores):.4f}")
        except Exception as e:
            print(f"Error training {name} model: {e}")
    
    if not results:
        print("No models could be successfully trained")
        return
    
    # Save detailed results
    with open(os.path.join(output_dir, 'regression_results.txt'), 'w') as f:
        for name, res in results.items():
            f.write(f"{name} results:\n")
            f.write(f"  MSE: {res['mse']:.4f}\n")
            f.write(f"  R²: {res['r2']:.4f}\n")
            f.write(f"  CV R² scores: {res['cv_scores']}\n")
            f.write(f"  Mean CV R²: {np.mean(res['cv_scores']):.4f}\n\n")
    
    # Compare model performance
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[m]['mse'] for m in results],
        'R²': [results[m]['r2'] for m in results],
        'Mean CV R²': [np.mean(results[m]['cv_scores']) for m in results]
    })
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='R²', data=model_comparison)
    plt.title('Model Comparison (R²)')
    plt.ylim(0, max(1.0, model_comparison['R²'].max() * 1.1))
    plt.savefig(os.path.join(output_dir, 'model_comparison_r2.png'))
    plt.close()
    
    # Use best model (by R²) for further analysis
    best_model_name = model_comparison.loc[model_comparison['R²'].idxmax(), 'Model']
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
    
    # Feature importance analysis - handle the pipeline structure
    if hasattr(best_model[-1], 'coef_'):
        # The model is inside the pipeline, so we need to be careful with feature names
        # Get feature names from the preprocessor
        try:
            feature_names = []
            
            # Get one-hot encoded column names for categorical features
            if categorical_features:
                cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
                encoded_features = []
                
                # Extract the categories from the encoder after it's been fit
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    # Newer scikit-learn versions
                    encoded_features = cat_encoder.get_feature_names_out(categorical_features)
                else:
                    # Older scikit-learn versions
                    encoded_features = [f"{col}_{cat}" for i, col in enumerate(categorical_features) 
                                        for cat in cat_encoder.categories_[i]]
                
                feature_names.extend(encoded_features)
            
            # Add numeric features
            feature_names.extend(numeric_features)
            
            # Now match with coefficients
            coeffs = pd.DataFrame({
                'Feature': feature_names[:len(best_model[-1].coef_)] if len(feature_names) >= len(best_model[-1].coef_) else feature_names + ['Unknown'] * (len(best_model[-1].coef_) - len(feature_names)),
                'Coefficient': best_model[-1].coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=coeffs.head(20))  # Show top 20 features
            plt.title(f'Feature Coefficients ({best_model_name})')
            plt.axvline(x=0, color='gray', linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_coefficients.png'))
            plt.close()
            
            # Save coefficients
            coeffs.to_csv(os.path.join(output_dir, 'feature_coefficients.csv'), index=False)
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, results[best_model_name]['y_pred'], alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted {target_col} ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # Plot residuals
    residuals = y_test - results[best_model_name]['y_pred']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results[best_model_name]['y_pred'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()
    
    # Plot residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'))
    plt.close()
    
    # Feature correlation analysis - focus only on numeric features
    correlation_data = df[numeric_features + [regression_target]].copy()
    
    # For the top 3 most correlated numeric features, create scatter plots
    try:
        correlation = correlation_data.corr()[regression_target].drop(regression_target)
        
        # Plot correlation for numeric features
        plt.figure(figsize=(12, 8))
        correlation_sorted = correlation.sort_values(ascending=False)
        sns.barplot(x=correlation_sorted.values, y=correlation_sorted.index)
        plt.title(f'Feature Correlation with {target_col}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
        plt.close()
        
        top_features = correlation.abs().sort_values(ascending=False).index[:min(3, len(correlation))]
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature], y=df[regression_target], alpha=0.5)
            plt.title(f'{feature} vs {target_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature}_scatter.png'))
            plt.close()
    except Exception as e:
        print(f"Error generating correlation analysis: {e}")
    
    # Write analysis of results
    with open(os.path.join(output_dir, 'analysis.txt'), 'w') as f:
        f.write(f"Linear Regression Analysis for {target_col.capitalize()} Prediction\n")
        f.write("=" * len(f"Linear Regression Analysis for {target_col.capitalize()} Prediction") + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of samples: {len(df)}\n")
        f.write(f"Target: {target_col}")
        if regression_target == 'log_target':
            f.write(" (log-transformed)\n")
        else:
            f.write("\n")
        
        f.write(f"Numeric features: {numeric_features}\n")
        f.write(f"Categorical features: {categorical_features}\n\n")
        
        f.write("Model Performance:\n")
        for name, res in results.items():
            f.write(f"  {name}:\n")
            f.write(f"    MSE: {res['mse']:.4f}\n")
            f.write(f"    R²: {res['r2']:.4f}\n")
            f.write(f"    Mean CV R²: {np.mean(res['cv_scores']):.4f}\n\n")
        
        f.write(f"Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})\n\n")
        
        if 'coeffs' in locals():
            f.write("Top 10 Most Important Features (by absolute coefficient):\n")
            for idx, row in coeffs.head(10).iterrows():
                feature = row['Feature']
                coef = row['Coefficient']
                f.write(f"  {feature}: {coef:.4f}\n")
            
            f.write("\nFeature Impact Analysis:\n")
            for idx, row in coeffs.head(15).iterrows():  # Limit to top 15 for readability
                feature = row['Feature']
                coef = row['Coefficient']
                
                if coef > 0:
                    impact = f"Positive impact: Higher values -> Higher {target_col}"
                else:
                    impact = f"Negative impact: Higher values -> Lower {target_col}"
                
                f.write(f"  {feature}:\n")
                f.write(f"    Coefficient: {coef:.4f}\n")
                f.write(f"    {impact}\n\n")
    
    print(f"\nLinear regression analysis complete. Results saved to {output_dir}")
    
    return best_model, model_comparison

if __name__ == "__main__":
    linear_regression_analysis()
