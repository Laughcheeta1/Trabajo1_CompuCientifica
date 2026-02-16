import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_basic_stats(df):
    """
    Returns basic statistics of the DataFrame.
    """
    return df.describe()

def check_missing_values_viz(df):
    """
    Plots missing values using a heatmap-like displot.
    """
    plt.figure(figsize=(10, 6))
    df.isnull().melt().pipe(lambda d: sns.displot(data=d, y='variable', hue='value', multiple='fill', aspect=2))
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_distributions_numerical(df, numerical_cols):
    """
    Plots distribution plots (hist + kde) for numerical columns.
    """
    num_cols = len(numerical_cols)
    rows = (num_cols // 3) + (1 if num_cols % 3 != 0 else 0)
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, numerical_cols):
    """
    Plots boxplots for numerical columns to identify outliers.
    """
    num_cols = len(numerical_cols)
    rows = (num_cols // 3) + (1 if num_cols % 3 != 0 else 0)
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_pie_categorical(df, categorical_cols):
    """
    Plots pie charts for categorical columns.
    """
    num_cols = len(categorical_cols)
    rows = (num_cols // 2) + (1 if num_cols % 2 != 0 else 0)
    
    fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 5))
    if rows > 1:
        axes = axes.flatten()
    elif num_cols == 1:
        axes = [axes] # Handle single plot case
    else: # rows=1, cols=2
        axes = axes # already list-like
        
    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            # Count values
            counts = df[col].value_counts()
            axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide empty subplots
    if num_cols < len(axes):
        for j in range(num_cols, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """
    Plots the correlation matrix of numerical columns.
    """
    plt.figure(figsize=(15, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
