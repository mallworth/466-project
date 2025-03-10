import pandas as pd
import numpy as np
import os

def safe_read_csv(file_path):
    """Enhanced CSV reading with error handling"""
    try:
        # Read the CSV file, skipping the first row (header) which contains the dataset name
        df = pd.read_csv(file_path, header=1, low_memory=False)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def preprocess_datasets():
    """Preprocess all datasets"""
    data_dir = './data'
    processed_dir = './data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Ensure detailed logging
    print("Starting dataset preprocessing...")
    
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
        
        file_path = os.path.join(data_dir, file)
        df = safe_read_csv(file_path)
        
        if df is None:
            print(f"Skipping {file} due to reading error")
            continue
        
        filename = file.replace('.csv', '')
        print(f"\nProcessing {filename}...")
        
        # Print initial dataset info
        print(f"Initial dataset shape: {df.shape}")
        print(f"Initial columns: {list(df.columns)}")
        
        # Handle percentage columns
        percentage_cols = [col for col in df.columns if col.endswith('_%')]
        for col in percentage_cols:
            new_col_name = col.replace('_%', '')
            try:
                # Convert percentage columns to decimal
                df[new_col_name] = df[col] / 100
                
                # Drop the original percentage column
                df = df.drop(columns=[col])
            except Exception as e:
                print(f"Error processing {col}: {e}")
        
        # Handle missing values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        # Additional preprocessing based on dataset
        if 'streams' in df.columns:
            # Ensure streams is numeric
            df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        
        if 'popularity' in df.columns:
            # Create popularity class
            median_popularity = df['popularity'].median()
            df['popularity_class'] = (df['popularity'] > median_popularity).astype(int)
        
        # Audio feature normalization
        audio_features = ['danceability', 'energy', 'acousticness', 
                          'instrumentalness', 'liveness', 'valence', 'speechiness']
        
        for feature in audio_features:
            if feature in df.columns:
                # Ensure feature is between 0 and 1
                if df[feature].max() > 1:
                    df[feature] = df[feature] / 100
                
                # Create categorical version
                df[f'{feature}_category'] = pd.cut(df[feature], 
                                                   bins=3, 
                                                   labels=['Low', 'Medium', 'High'])
        
        # Create processed dataset
        output_path = os.path.join(processed_dir, f"{filename}_processed.csv")
        df.to_csv(output_path, index=False)
        
        # Print processed dataset info
        print(f"Processed dataset shape: {df.shape}")
        print(f"Processed columns: {list(df.columns)}")
        print(f"Saved processed dataset to {output_path}")

def main():
    preprocess_datasets()

if __name__ == "__main__":
    main()
