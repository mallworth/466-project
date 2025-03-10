import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import glob
import sys

def safe_read_csv(file_path):
    """Enhanced CSV reading with error handling"""
    try:
        # Try the most basic reading first
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully read {file_path} with standard reading")
        return df
    except Exception as e:
        print(f"Error with standard reading of {file_path}: {e}")
        try:
            # Try reading with different encoding
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            print(f"Successfully read {file_path} with latin1 encoding")
            return df
        except Exception as e2:
            print(f"Error with latin1 encoding: {e2}")
            try:
                # Try skipping the first row
                df = pd.read_csv(file_path, header=1, low_memory=False)
                print(f"Successfully read {file_path} by skipping first row")
                return df
            except Exception as e3:
                print(f"All reading methods failed for {file_path}: {e3}")
                return None

def print_file_preview(file_path):
    """Print the first few lines of a file for debugging"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            head = [next(f) for _ in range(5)]
        print("\nFile preview (first 5 lines):")
        for i, line in enumerate(head):
            print(f"Line {i+1}: {line.strip()}")
    except Exception as e:
        try:
            with open(file_path, 'r', encoding='latin1') as f:
                head = [next(f) for _ in range(5)]
            print("\nFile preview (first 5 lines, latin1 encoding):")
            for i, line in enumerate(head):
                print(f"Line {i+1}: {line.strip()}")
        except Exception as e2:
            print(f"Could not read file preview: {e2}")

def decision_tree_classification():
    """Apply decision tree classifier to predict music genre"""
    print("Starting decision tree classification...")
    
    data_dir = './data'
    processed_dir = './data/processed'
    output_dir = './results/decision_tree'
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for CSV files in all directories
    search_dirs = [data_dir]
    if os.path.exists(processed_dir):
        search_dirs.append(processed_dir)
    
    print("\nSearching for datasets in:")
    for dir_path in search_dirs:
        print(f"- {dir_path}")
        if not os.path.exists(dir_path):
            print(f"  WARNING: Directory {dir_path} does not exist!")
    
    all_csv_files = []
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            all_csv_files.extend(csv_files)
    
    print(f"\nFound {len(all_csv_files)} CSV files:")
    for csv_file in all_csv_files:
        print(f"- {os.path.basename(csv_file)}")
    
    if not all_csv_files:
        print("No CSV files found. Please ensure your data files are in the right location.")
        return
    
    # Let's try each file until we find one we can use
    usable_dataset = None
    
    for file_path in all_csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*40}")
        print(f"Examining file: {file_name}")
        print(f"{'='*40}")
        
        # Read the file
        df = safe_read_csv(file_path)
        if df is None:
            print("Skipping this file due to reading errors.")
            continue
        
        # Print column names and basic info
        print(f"\nDataframe shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # If we have only one column, it might be a malformed CSV
        if len(df.columns) == 1:
            column_name = df.columns[0]
            first_value = df.iloc[0, 0] if not df.empty else "N/A"
            print(f"\nWARNING: Only one column detected: '{column_name}'")
            print(f"First value: '{first_value}'")
            print("This might indicate CSV parsing issues. Checking file format...")
            print_file_preview(file_path)
            
            # Try to manually check for delimiter
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            print(f"\nFirst line of file: {first_line}")
            
            # Check potential delimiters
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                count = first_line.count(delimiter)
                if count > 0:
                    print(f"Potential delimiter '{delimiter}' found {count} times")
                    try:
                        df_retry = pd.read_csv(file_path, sep=delimiter, low_memory=False)
                        print(f"Successfully read with delimiter '{delimiter}'!")
                        print(f"New columns: {df_retry.columns.tolist()}")
                        df = df_retry
                        break
                    except Exception as e:
                        print(f"Failed with delimiter '{delimiter}': {e}")
        
        # Check for genre column
        genre_column = None
        for col in df.columns:
            if 'genre' in col.lower():
                genre_column = col
                print(f"\nFound genre column: '{genre_column}'")
                break
        
        if genre_column is None:
            print("No genre column found in this file. Skipping.")
            continue
        
        # Check for audio features
        standard_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                             'speechiness', 'acousticness', 'instrumentalness', 
                             'liveness', 'valence', 'tempo', 'duration_ms']
        
        found_features = []
        for feature in standard_features:
            for col in df.columns:
                if feature.lower() in col.lower():
                    found_features.append(col)
                    print(f"Found feature: '{col}' matching '{feature}'")
                    break
        
        # If no standard features found, check if there are other numeric columns we can use
        if len(found_features) < 3:
            print("\nNot enough standard audio features found. Checking for any numeric columns...")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != genre_column]
            
            if len(numeric_cols) >= 3:
                print(f"Found {len(numeric_cols)} numeric columns to use as features:")
                for col in numeric_cols[:10]:  # Show just the first 10 if there are many
                    print(f"- {col}")
                found_features = numeric_cols
                
        if len(found_features) >= 3:
            print(f"\nFile {file_name} has genre column and {len(found_features)} usable features!")
            usable_dataset = {
                'file_path': file_path,
                'file_name': file_name,
                'dataframe': df,
                'genre_column': genre_column,
                'features': found_features
            }
            break
        else:
            print(f"Not enough features found in {file_name}. Skipping.")
    
    if usable_dataset is None:
        print("\nNo usable dataset found with both genre information and sufficient features.")
        return
    
    # Use the selected dataset
    df = usable_dataset['dataframe']
    genre_column = usable_dataset['genre_column']
    features = usable_dataset['features']
    
    print(f"\n{'='*40}")
    print(f"Using dataset: {usable_dataset['file_name']}")
    print(f"Genre column: {genre_column}")
    print(f"Features: {features}")
    print(f"{'='*40}")
    
    # Ensure features are numeric
    for feature in features[:]:  # Create a copy of the list to modify during iteration
        if not pd.api.types.is_numeric_dtype(df[feature]):
            try:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                print(f"Converted {feature} to numeric")
            except:
                print(f"Removing {feature} as it cannot be converted to numeric")
                features.remove(feature)
    
    if len(features) < 3:
        print("Not enough numeric features for meaningful classification")
        return
    
    # Count genres
    genre_counts = df[genre_column].value_counts()
    print("\nGenre distribution:")
    print(genre_counts.head(10))  # Show just the top 10 if there are many
    
    # Plot genre distribution
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(genre_counts))
    sns.barplot(y=genre_counts.index[:top_n], x=genre_counts.values[:top_n])
    plt.title(f'Distribution of Top {top_n} Genres')
    plt.xlabel('Number of Tracks')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_distribution.png'))
    plt.close()
    
    # Restrict to genres with enough samples
    min_samples = 20
    valid_genres = genre_counts[genre_counts >= min_samples].index
    
    if len(valid_genres) < 2:
        min_samples = 10  # Try with fewer samples
        valid_genres = genre_counts[genre_counts >= min_samples].index
        print(f"Reduced minimum samples to {min_samples} to get more genres")
    
    df_filtered = df[df[genre_column].isin(valid_genres)]
    
    # Handle the case where all genres have fewer than min_samples
    if len(df_filtered) < 100 or len(valid_genres) < 2:
        top_n = max(2, min(5, len(genre_counts)))
        print(f"Not enough genres with {min_samples} samples. Using top {top_n} genres instead.")
        top_genres = genre_counts.nlargest(top_n).index
        df_filtered = df[df[genre_column].isin(top_genres)]
    
    print(f"Using {len(df_filtered)} samples from {len(df_filtered[genre_column].unique())} genres")
    
    # Prepare data
    X = df_filtered[features].copy()
    y = df_filtered[genre_column].copy()
    
    # Fill missing values
    X = X.fillna(X.median())
    
    # Encode genre labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save encoding mapping
    encoding_map = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    with open(os.path.join(output_dir, 'genre_encoding.txt'), 'w') as f:
        for genre, code in encoding_map.items():
            f.write(f"{genre}: {code}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    # Simplified parameter grid for faster execution
    param_grid = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    print("\nPerforming grid search to find optimal parameters...")
    tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(tree, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print("\nBest parameters:", best_params)
    
    # Save best parameters
    with open(os.path.join(output_dir, 'best_parameters.txt'), 'w') as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    # Train model with best parameters
    best_tree = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_tree.predict(X_test)
    
    # Evaluate model
    accuracy = grid_search.best_score_
    print(f"\nModel accuracy (CV): {accuracy:.4f}")
    
    # Test set accuracy
    test_accuracy = best_tree.score(X_test, y_test)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Convert back to original labels for the report
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    report = classification_report(y_test_decoded, y_pred_decoded)
    
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model accuracy (CV): {accuracy:.4f}\n")
        f.write(f"Test set accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    except Exception as e:
        print(f"Error creating confusion matrix visualization: {e}")
        # Simplified version without labels if there are too many classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_simple.png'))
    plt.close()
    
    # Feature importance
    importance = best_tree.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Genre Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    # Visualize the decision tree (limit depth to make it readable)
    plt.figure(figsize=(20, 10))
    max_depth_to_plot = min(3, best_tree.tree_.max_depth)
    
    try:
        plot_tree(best_tree, max_depth=max_depth_to_plot, feature_names=features, 
                class_names=label_encoder.classes_, filled=True, rounded=True, fontsize=10)
        plt.title(f'Decision Tree for Music Genre Classification (Limited to Depth {max_depth_to_plot})')
        plt.savefig(os.path.join(output_dir, 'decision_tree.png'))
    except Exception as e:
        print(f"Error creating decision tree visualization: {e}")
        # Try with indices instead of class names
        plot_tree(best_tree, max_depth=max_depth_to_plot, feature_names=features, 
                filled=True, rounded=True, fontsize=10)
        plt.title(f'Decision Tree for Music Genre Classification (Limited to Depth {max_depth_to_plot})')
        plt.savefig(os.path.join(output_dir, 'decision_tree.png'))
    plt.close()
    
    print(f"\nDecision tree analysis complete. Results saved to {output_dir}")
    
    return best_tree, feature_importance

if __name__ == "__main__":
    decision_tree_classification()
