import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
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

def association_rule_mining():
    """Discover association rules in music features"""
    data_dir = './data'
    processed_dir = './data/processed'
    output_dir = './results/association_rules'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find suitable datasets in both raw and processed directories
    datasets = {}
    
    # Check processed directory first if it exists
    if os.path.exists(processed_dir):
        for file in os.listdir(processed_dir):
            if file.endswith('.csv'):
                name = file.replace('_processed.csv', '').replace('.csv', '')
                file_path = os.path.join(processed_dir, file)
                df = safe_read_csv(file_path)
                if df is not None:
                    datasets[name] = df
    
    # If no processed datasets or they don't have enough features, check raw data
    if not datasets or not any('danceability' in df.columns for df in datasets.values()):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                name = file.replace('.csv', '')
                file_path = os.path.join(data_dir, file)
                df = safe_read_csv(file_path)
                if df is not None:
                    datasets[name] = df
    
    if not datasets:
        print("No datasets found")
        return
    
    # Use the largest dataset that has audio features
    selected_dataset = None
    for name, df in datasets.items():
        # Check for various spellings/formats of audio features
        audio_feature_variants = {
            'danceability': ['danceability', 'danceability_%', 'dance'],
            'energy': ['energy', 'energy_%', 'nrg'],
            'loudness': ['loudness', 'loud'],
            'speechiness': ['speechiness', 'speechiness_%', 'speech'],
            'acousticness': ['acousticness', 'acousticness_%', 'acoustic'],
            'instrumentalness': ['instrumentalness', 'instrumentalness_%'],
            'liveness': ['liveness', 'liveness_%', 'live'],
            'valence': ['valence', 'valence_%', 'val'],
            'tempo': ['tempo', 'bpm']
        }
        
        # Count how many audio feature types are present
        feature_count = 0
        found_features = []
        
        for feature, variants in audio_feature_variants.items():
            for variant in variants:
                if variant in df.columns:
                    feature_count += 1
                    found_features.append(variant)
                    break
        
        if feature_count >= 3:  # Need at least 3 audio features
            if selected_dataset is None or len(df) > len(datasets[selected_dataset]):
                selected_dataset = name
                print(f"Found candidate dataset: {name} with {feature_count} audio features")
    
    if selected_dataset is None:
        print("No suitable dataset found with audio features")
        return
    
    df = datasets[selected_dataset]
    print(f"Using {selected_dataset} dataset with {len(df)} samples")
    
    # Create binary features for association rule mining
    binary_features = []
    
    # Expanded list of audio features to check for
    all_audio_features = [
        'danceability', 'energy', 'loudness', 'acousticness', 'speechiness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
        'danceability_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 
        'liveness_%', 'valence_%', 'speechiness_%', 'bpm', 'duration'
    ]
    
    # Keep only features that exist in the dataframe
    features = [f for f in all_audio_features if f in df.columns]
    
    if len(features) < 3:
        print("Not enough audio features for meaningful association rule mining")
        return
    
    print(f"Using {len(features)} features: {features}")
    
    # Discretize each feature into categories (Low, Medium, High)
    for feature in features:
        # Skip if not numeric or already processed
        if not pd.api.types.is_numeric_dtype(df[feature]) or f'{feature}_Low' in binary_features:
            continue
            
        # Create categorical feature
        try:
            # Use qcut for equal-sized bins, with handling for duplicate values
            df[f'{feature}_cat'] = pd.qcut(df[feature].rank(method='first'), 
                                      q=3, 
                                      labels=['Low', 'Medium', 'High'])
            
            # Create binary features
            for category in ['Low', 'Medium', 'High']:
                col_name = f'{feature}_{category}'
                df[col_name] = (df[f'{feature}_cat'] == category).astype(int)
                binary_features.append(col_name)
        except Exception as e:
            print(f"Could not discretize {feature}: {e}")
    
    # Check for various genre column names
    genre_columns = [col for col in df.columns if 'genre' in col.lower()]
    
    if genre_columns:
        genre_col = genre_columns[0]
        print(f"Found genre column: {genre_col}")
        
        # Get top genres
        top_genres = df[genre_col].value_counts().nlargest(10).index
        
        for genre in top_genres:
            col_name = f'genre_{genre}'
            df[col_name] = (df[genre_col] == genre).astype(int)
            binary_features.append(col_name)
    
    # Check for popularity/streams information
    popularity_cols = ['popularity', 'streams', 'popularity_percent', 'stream_count']
    
    for col in popularity_cols:
        if col in df.columns:
            try:
                # Convert to numeric if needed
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Create categories
                df[f'{col}_cat'] = pd.qcut(df[col].rank(method='first'), 
                                     q=3, 
                                     labels=['Low', 'Medium', 'High'])
                
                for category in ['Low', 'Medium', 'High']:
                    col_name = f'{col}_{category}'
                    df[col_name] = (df[f'{col}_cat'] == category).astype(int)
                    binary_features.append(col_name)
            except Exception as e:
                print(f"Could not discretize {col}: {e}")
    
    print(f"Created {len(binary_features)} binary features for association rule mining")
    
    if len(binary_features) < 3:
        print("Not enough binary features created for association rule mining")
        return
    
    # Apply Apriori algorithm to find frequent itemsets
    binary_df = df[binary_features].fillna(0)
    
    # Convert to boolean for better performance in apriori
    binary_df_bool = binary_df.astype(bool)
    
    # Save binary dataframe for reference
    binary_df.to_csv(os.path.join(output_dir, 'binary_features.csv'), index=False)
    
    # Find frequent itemsets with decreasing support until we find some
    supports = [0.1, 0.05, 0.03, 0.02, 0.01]
    frequent_itemsets = None
    
    for min_support in supports:
        print(f"Trying min_support={min_support}")
        try:
            frequent_itemsets = apriori(binary_df_bool, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")
                break
            else:
                print("No frequent itemsets found with this support level")
        except Exception as e:
            print(f"Error with min_support={min_support}: {e}")
    
    if frequent_itemsets is None or len(frequent_itemsets) == 0:
        print("No frequent itemsets found with any support level")
        return
    
    # Save frequent itemsets
    frequent_itemsets.to_csv(os.path.join(output_dir, 'frequent_itemsets.csv'), index=False)
    
    # Generate association rules with decreasing confidence until we find some
    confidences = [0.7, 0.5, 0.3, 0.2]
    rules = None
    
    for min_confidence in confidences:
        print(f"Trying min_confidence={min_confidence}")
        try:
            rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
            
            if len(rules) > 0:
                print(f"Found {len(rules)} association rules with min_confidence={min_confidence}")
                break
            else:
                print("No rules found with this confidence level")
        except Exception as e:
            print(f"Error with min_confidence={min_confidence}: {e}")
    
    if rules is None or len(rules) == 0:
        print("No association rules found with any confidence level")
        return
    
    # Save all rules
    rules.to_csv(os.path.join(output_dir, 'association_rules.csv'), index=False)
    
    # Sort rules by lift
    rules = rules.sort_values('lift', ascending=False)
    
    # Save top rules
    top_rules = rules.head(20)
    top_rules.to_csv(os.path.join(output_dir, 'top_rules_by_lift.csv'), index=False)
    
    # Convert rule format for better readability
    readable_rules = []
    for idx, row in top_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        antecedents_str = ', '.join([str(item) for item in antecedents])
        consequents_str = ', '.join([str(item) for item in consequents])
        
        rule_dict = {
            'Rule': f"{idx+1}",
            'Antecedents': antecedents_str,
            'Consequents': consequents_str,
            'Support': row['support'],
            'Confidence': row['confidence'],
            'Lift': row['lift']
        }
        readable_rules.append(rule_dict)
    
    # Save readable rules
    with open(os.path.join(output_dir, 'top_rules.txt'), 'w') as f:
        f.write(f"Top {len(readable_rules)} association rules by lift:\n\n")
        for rule in readable_rules:
            f.write(f"Rule {rule['Rule']}:\n")
            f.write(f"  IF {rule['Antecedents']}\n")
            f.write(f"  THEN {rule['Consequents']}\n")
            f.write(f"  Support: {rule['Support']:.4f}\n")
            f.write(f"  Confidence: {rule['Confidence']:.4f}\n")
            f.write(f"  Lift: {rule['Lift']:.4f}\n\n")
    
    # Visualize top rules
    plt.figure(figsize=(10, 8))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=rules['lift']*20)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules (size represents lift)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rules_scatter.png'))
    plt.close()
    
    # Plot top rules by lift
    top_rules_df = pd.DataFrame(readable_rules)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Lift', y='Rule', data=top_rules_df)
    plt.title('Top Rules by Lift')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_rules_lift.png'))
    plt.close()
    
    # Plot top rules by confidence
    top_by_confidence = rules.sort_values('confidence', ascending=False).head(20)
    top_by_confidence_readable = []
    
    for idx, row in top_by_confidence.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        antecedents_str = ', '.join([str(item) for item in antecedents])
        consequents_str = ', '.join([str(item) for item in consequents])
        
        rule_dict = {
            'Rule': f"{idx+1}",
            'Antecedents': antecedents_str,
            'Consequents': consequents_str,
            'Support': row['support'],
            'Confidence': row['confidence'],
            'Lift': row['lift']
        }
        top_by_confidence_readable.append(rule_dict)
    
    top_by_confidence_df = pd.DataFrame(top_by_confidence_readable)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Confidence', y='Rule', data=top_by_confidence_df)
    plt.title('Top Rules by Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_rules_confidence.png'))
    plt.close()
    
    # Generate insights from rules
    with open(os.path.join(output_dir, 'rule_insights.txt'), 'w') as f:
        f.write("Insights from Association Rules:\n\n")
        
        # Check for genre-related rules
        genre_rules = rules[rules['antecedents'].apply(lambda x: any('genre_' in str(item) for item in x)) | 
                           rules['consequents'].apply(lambda x: any('genre_' in str(item) for item in x))]
        
        if len(genre_rules) > 0:
            f.write("Genre-related patterns:\n")
            top_genre_rules = genre_rules.sort_values('lift', ascending=False).head(5)
            
            for idx, row in top_genre_rules.iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                
                antecedents_str = ', '.join([str(item) for item in antecedents])
                consequents_str = ', '.join([str(item) for item in consequents])
                
                f.write(f"  Rule: IF {antecedents_str} THEN {consequents_str}\n")
                f.write(f"  Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f}\n\n")
        
        # Check for popularity-related rules
        popularity_rules = rules[rules['antecedents'].apply(lambda x: any(('popularity_' in str(item) or 'streams_' in str(item)) for item in x)) | 
                                rules['consequents'].apply(lambda x: any(('popularity_' in str(item) or 'streams_' in str(item)) for item in x))]
        
        if len(popularity_rules) > 0:
            f.write("Popularity-related patterns:\n")
            top_popularity_rules = popularity_rules.sort_values('lift', ascending=False).head(5)
            
            for idx, row in top_popularity_rules.iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                
                antecedents_str = ', '.join([str(item) for item in antecedents])
                consequents_str = ', '.join([str(item) for item in consequents])
                
                f.write(f"  Rule: IF {antecedents_str} THEN {consequents_str}\n")
                f.write(f"  Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f}\n\n")
        
        # Add general patterns section
        f.write("General audio feature patterns:\n")
        general_rules = rules.sort_values('lift', ascending=False).head(10)
        
        for idx, row in general_rules.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            antecedents_str = ', '.join([str(item) for item in antecedents])
            consequents_str = ', '.join([str(item) for item in consequents])
            
            f.write(f"  Rule: IF {antecedents_str} THEN {consequents_str}\n")
            f.write(f"  Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f}\n\n")
    
    print(f"\nAssociation rule mining complete. Results saved to {output_dir}")
    return rules

if __name__ == "__main__":
    association_rule_mining()
