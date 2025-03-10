import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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

def naive_bayes_classification():
    """Apply Naive Bayes classifier to predict music popularity"""
    data_dir = './data'
    processed_dir = './data/processed'
    output_dir = './results/naive_bayes'
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for datasets with popularity information
    datasets = {}
    
    # Check both raw and processed directories
    search_dirs = [data_dir]
    if os.path.exists(processed_dir):
        search_dirs.append(processed_dir)
    
    # Try to merge high/low popularity datasets if they exist
    high_pop_path = os.path.join(processed_dir, 'high_popularity_spotify_data_processed.csv')
    low_pop_path = os.path.join(processed_dir, 'low_popularity_spotify_data_processed.csv')
    
    if os.path.exists(high_pop_path) and os.path.exists(low_pop_path):
        high_df = safe_read_csv(high_pop_path)
        low_df = safe_read_csv(low_pop_path)
        
        if high_df is not None and low_df is not None:
            # Add labels
            high_df['popularity_class'] = 1  # High popularity
            low_df['popularity_class'] = 0   # Low popularity
            
            # Combine datasets
            combined_df = pd.concat([high_df, low_df], ignore_index=True)
            datasets['combined_popularity'] = combined_df
            
            print(f"Created combined popularity dataset with {len(combined_df)} samples")
    
    # Look for datasets with popularity or streams columns
    for dir_path in search_dirs:
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                file_path = os.path.join(dir_path, file)
                name = file.replace('_processed.csv', '').replace('.csv', '')
                
                # Use safe_read_csv to handle dataset title in first line
                df = safe_read_csv(file_path)
                
                if df is None:
                    continue
                
                # Check for popularity column
                if 'popularity' in df.columns:
                    try:
                        # Try to convert to numeric
                        if not pd.api.types.is_numeric_dtype(df['popularity']):
                            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
                        
                        # Drop rows with missing popularity
                        df = df.dropna(subset=['popularity'])
                        
                        # Create binary class based on median
                        median_popularity = df['popularity'].median()
                        df['popularity_class'] = (df['popularity'] > median_popularity).astype(int)
                        
                        datasets[name] = df
                        print(f"Using {name} dataset with {len(df)} samples")
                    except Exception as e:
                        print(f"Error processing popularity in {name}: {e}")
                
                # Check for streams column if popularity not found
                elif 'streams' in df.columns and name not in datasets:
                    try:
                        # Convert streams to numeric if needed
                        if not pd.api.types.is_numeric_dtype(df['streams']):
                            df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
                        
                        # Drop rows with missing streams
                        df = df.dropna(subset=['streams'])
                        
                        # Create binary class based on median
                        median_streams = df['streams'].median()
                        df['popularity_class'] = (df['streams'] > median_streams).astype(int)
                        
                        datasets[name] = df
                        print(f"Using {name} dataset with streams as popularity indicator ({len(df)} samples)")
                    except Exception as e:
                        print(f"Error processing streams in {name}: {e}")
    
    if not datasets:
        print("No suitable datasets found with popularity information")
        return
    
    # Use the dataset with the most samples
    dataset_name = max(datasets.keys(), key=lambda k: len(datasets[k]))
    df = datasets[dataset_name].copy()
    
    # For large datasets, sample to speed up processing
    if len(df) > 10000:
        print(f"Sampling 10,000 rows from {len(df)} samples for faster processing")
        df = df.sample(10000, random_state=42)
    
    print(f"Selected {dataset_name} with {len(df)} samples for Naive Bayes classification")
    
    # Select features
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo', 'duration_ms']
    
    # Keep only features that exist in the dataframe
    features = [f for f in audio_features if f in df.columns]
    
    if len(features) < 3:
        print("Not enough audio features for meaningful classification")
        return
    
    print(f"Using {len(features)} features: {features}")
    
    # Separate numeric and categorical features
    numeric_features = []
    categorical_features = []
    
    for feature in features:
        # Check if feature is categorical
        if pd.api.types.is_object_dtype(df[feature]) or feature in ['key', 'mode'] or feature == 'tempo':
            categorical_features.append(feature)
        else:
            try:
                # Try to convert to numeric
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                numeric_features.append(feature)
            except Exception as e:
                print(f"Error converting {feature} to numeric: {e}")
                categorical_features.append(feature)
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Prepare data
    X = df[features].copy()
    y = df['popularity_class']
    
    # Fill missing values
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown')
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ] if categorical_features else [('num', StandardScaler(), numeric_features)],
        remainder='passthrough'
    )
    
    # Create full pipeline with Naive Bayes
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores):.4f}")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate model
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model accuracy: {accuracy:.4f}\n\n")
        f.write(f"Cross-validation scores: {cv_scores}\n")
        f.write(f"Mean CV score: {np.mean(cv_scores):.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC curve
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating ROC curve: {e}")
    
    # Feature importance analysis
    # For this version we'll focus on numeric features that we can analyze directly
    if len(numeric_features) >= 3:
        # Create a simpler model with just numeric features to analyze importance
        X_numeric = df[numeric_features].copy().fillna(0)
        X_numeric_scaled = StandardScaler().fit_transform(X_numeric)
        
        # Train a simple GaussianNB on numeric features only
        simple_nb = GaussianNB()
        simple_nb.fit(X_numeric_scaled, y)
        
        # Check which attributes are available for this version of scikit-learn
        # In newer versions, 'sigma_' is replaced with 'var_'
        theta_attr = 'theta_'  # Means of features per class
        sigma_attr = 'var_' if hasattr(simple_nb, 'var_') else 'sigma_'  # Variances of features per class
        
        # Analyze feature impact on class probabilities
        feature_impact = pd.DataFrame({
            'Feature': numeric_features,
            'Class 0 Mean': [simple_nb.theta_[0, i] for i in range(len(numeric_features))],
            'Class 1 Mean': [simple_nb.theta_[1, i] for i in range(len(numeric_features))],
            'Class 0 Variance': [getattr(simple_nb, sigma_attr)[0, i] for i in range(len(numeric_features))],
            'Class 1 Variance': [getattr(simple_nb, sigma_attr)[1, i] for i in range(len(numeric_features))]
        })
        
        # Calculate difference in means (impact)
        feature_impact['Mean Difference'] = abs(feature_impact['Class 1 Mean'] - feature_impact['Class 0 Mean'])
        feature_impact = feature_impact.sort_values('Mean Difference', ascending=False)
        
        # Save feature impact details
        feature_impact.to_csv(os.path.join(output_dir, 'feature_impact.csv'), index=False)
        
        # Plot feature impact
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Mean Difference', y='Feature', data=feature_impact)
        plt.title('Feature Impact on Popularity Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_impact.png'))
        plt.close()
        
        # Plot distribution of top features by impact
        top_features = feature_impact.head(3)['Feature'].values
        
        for feature in top_features:
            plt.figure(figsize=(12, 6))
            feature_idx = numeric_features.index(feature)
            
            # Distribution for class 0
            mean_0 = simple_nb.theta_[0, feature_idx]
            var_0 = getattr(simple_nb, sigma_attr)[0, feature_idx]
            
            # Check for very small variance to avoid division by zero
            if var_0 < 1e-10:
                var_0 = 1e-10
                
            x_0 = np.linspace(mean_0 - 3*np.sqrt(var_0), mean_0 + 3*np.sqrt(var_0), 100)
            y_0 = 1/(np.sqrt(2*np.pi*var_0)) * np.exp(-(x_0-mean_0)**2/(2*var_0))
            
            # Distribution for class 1
            mean_1 = simple_nb.theta_[1, feature_idx]
            var_1 = getattr(simple_nb, sigma_attr)[1, feature_idx]
            
            # Check for very small variance to avoid division by zero
            if var_1 < 1e-10:
                var_1 = 1e-10
                
            x_1 = np.linspace(mean_1 - 3*np.sqrt(var_1), mean_1 + 3*np.sqrt(var_1), 100)
            y_1 = 1/(np.sqrt(2*np.pi*var_1)) * np.exp(-(x_1-mean_1)**2/(2*var_1))
            
            plt.plot(x_0, y_0, 'b-', label='Low Popularity')
            plt.plot(x_1, y_1, 'r-', label='High Popularity')
            plt.title(f'Distribution of {feature} by Popularity Class')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
            plt.close()
        
        # Create a summary table of feature differences
        plt.figure(figsize=(14, 10))
        
        # Create a heatmap to show feature distributions by class
        feature_dist = pd.DataFrame()
        for i, feature in enumerate(numeric_features):
            feature_dist[f'{feature}_Low'] = [simple_nb.theta_[0, i]]
            feature_dist[f'{feature}_High'] = [simple_nb.theta_[1, i]]
        
        sns.heatmap(feature_dist, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Means by Popularity Class')
        plt.savefig(os.path.join(output_dir, 'feature_means_heatmap.png'))
        plt.close()
        
        # Write analysis of results
        with open(os.path.join(output_dir, 'analysis.txt'), 'w') as f:
            f.write("Naive Bayes Analysis for Popularity Prediction\n")
            f.write("============================================\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Number of samples: {len(df)}\n")
            f.write(f"Features used for model: {features}\n")
            f.write(f"Features used for analysis: {numeric_features}\n\n")
            
            f.write(f"Model accuracy: {accuracy:.4f}\n")
            
            if 'roc_auc' in locals():
                f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
            
            f.write("Top 3 most influential features:\n")
            for idx, row in feature_impact.head(3).iterrows():
                feature = row['Feature']
                diff = row['Mean Difference']
                
                feature_idx = numeric_features.index(feature)
                mean_0 = simple_nb.theta_[0, feature_idx]
                mean_1 = simple_nb.theta_[1, feature_idx]
                
                f.write(f"  {feature}:\n")
                f.write(f"    Mean Difference: {diff:.4f}\n")
                f.write(f"    Low Popularity Mean: {mean_0:.4f}\n")
                f.write(f"    High Popularity Mean: {mean_1:.4f}\n")
                f.write(f"    Indicates: {'Higher values -> Higher popularity' if mean_1 > mean_0 else 'Lower values -> Higher popularity'}\n\n")
    else:
        print("Not enough numeric features for detailed analysis")
        
        # Write a simpler analysis
        with open(os.path.join(output_dir, 'analysis.txt'), 'w') as f:
            f.write("Naive Bayes Analysis for Popularity Prediction\n")
            f.write("============================================\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Number of samples: {len(df)}\n")
            f.write(f"Features used: {features}\n\n")
            f.write(f"Model accuracy: {accuracy:.4f}\n")
            
            if 'roc_auc' in locals():
                f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    
    print(f"\nNaive Bayes analysis complete. Results saved to {output_dir}")
    
    return pipeline, None if 'feature_impact' not in locals() else feature_impact

if __name__ == "__main__":
    naive_bayes_classification()
