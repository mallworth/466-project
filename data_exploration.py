import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')

def explore_dataset(file_path):
    """Basic exploration of dataset with special handling for dataset title in first line"""
    filename = os.path.basename(file_path).replace('.csv', '')
    
    # Check the first few lines to determine file format
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f.readlines()[:5]]
    
    # Check if the first line is a dataset title (not a header)
    # This is detected if the second line contains many commas (column names)
    if len(lines) >= 2:
        if lines[0].count(',') <= 1 and lines[1].count(',') > 1:
            print(f"Detected dataset title in first line: '{lines[0]}'")
            # Skip the first line (dataset title) and read from the second line
            df = pd.read_csv(file_path, skiprows=1)
            print(f"Reading {filename} by skipping first line (dataset title)")
        else:
            # Standard CSV reading
            df = pd.read_csv(file_path)
            print(f"Reading {filename} with standard CSV format")
    else:
        # Not enough lines to determine, use standard reading
        df = pd.read_csv(file_path)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {filename}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Handle numeric description separately because some columns might not be numeric
    try:
        print("\nBasic statistics for numeric columns:")
        print(df.describe())
    except Exception as e:
        print(f"Could not generate statistics: {str(e)}")
    
    # Create visualizations directory for this dataset
    output_dir = os.path.join('./visualizations', filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize distributions of numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols[:10]:  # Limit to first 10 columns to avoid too many plots
        plt.figure(figsize=(10, 6))
        try:
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(output_dir, f'dist_{col}.png'))
        except Exception as e:
            print(f"Could not plot distribution for {col}: {str(e)}")
        plt.close()
    
    # Visualize categorical data
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
            plt.figure(figsize=(12, 8))
            try:
                top_cats = df[col].value_counts().nlargest(15)
                sns.barplot(y=top_cats.index, x=top_cats.values)
                plt.title(f'Top 15 values for {col}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'cat_{col}.png'))
            except Exception as e:
                print(f"Could not plot categories for {col}: {str(e)}")
            plt.close()
    
    # Create correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(14, 10))
        try:
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        except Exception as e:
            print(f"Could not create correlation matrix: {str(e)}")
        plt.close()
    
    return df

# Process each dataset
data_dir = './data'
datasets = {}

# Check if the data directory exists
if not os.path.exists(data_dir):
    print(f"Error: Data directory '{data_dir}' not found. Please create it and add your CSV files.")
    exit(1)

# List available datasets
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

if not csv_files:
    print(f"No CSV files found in '{data_dir}'. Please add your datasets to this directory.")
    exit(1)

print(f"Found {len(csv_files)} datasets:")
for i, file in enumerate(csv_files):
    print(f"{i+1}. {file}")

# Process each dataset
for file in csv_files:
    try:
        file_path = os.path.join(data_dir, file)
        datasets[file] = explore_dataset(file_path)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("\nData exploration complete. Visualizations saved to './visualizations' directory.")
