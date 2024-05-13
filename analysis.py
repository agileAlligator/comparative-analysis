import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_combined = {
    "File Name": ["nightshot_iso_100", "fireworks", "big_tree", "big_building", "cathedral", "spider_web", "deer", "leaves_iso_200", "hdr", "flower_foveon", "nightshot_iso_1600", "leaves_iso_1600", "bridge", "artificial"],
    "MultinomialBitsOver": [6.9e-190, "eps", "eps", "eps", 1.1e-5, "eps", 1.5e-6, 1.1e-13, 4.0e-280, 2.9e-295, 8.7e-24, 3.2e-4, 3.9e-24, "eps"],
    "ClosePairsBitMatch, t = 2": ["", 3.3e-70, 2.2e-8, 8.4e-68, "", 3.1e-4, "", "", "", "", "", "", "", 1.6e-70],
    "ClosePairsBitMatch, t = 4": ["", 1.8e-148, "", 4.5e-146, "", "", "", "", "", "", "", "", "", 8.7e-149],
    "AppearanceSpacings": ["", 1 - 1.0e-15, "", "", "", 1 - 1.0e-15, "", "", "", "", "", "", "", "1-eps1"],
    "LempelZiv": ["1 - 3.3e-6", 1 - 1.0e-15, 1 - 1.1e-5, 1 - 1.0e-15, "", 1 - 6.0e-5, "", "", "", "", 0.9997, "", "", "1-eps1"],
    "Fourier3": ["", 7.3e-7, "", 2.8e-5, "", "", "", "", "", "", "", "", "", 2.1e-18],
    "PeriodsInStrings": ["", "", "", "", "", 9.5e-10, "", "", "", "", "", "", "", "eps"],
    "HammingWeight": ["eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps"],
    "HammingCorr, L = 32": ["eps", "eps", 2.3e-12, "eps", 3.3e-14, "eps", 3.3e-8, 5.1e-12, "eps", "eps", 6.0e-12, 2.8e-6, 1.1e-9, "eps"],
    "HammingCorr, L = 64": ["eps", "eps", "eps", "eps", "", "eps", "eps", "eps", "eps", "eps", "eps", 1.1e-16, 1.1e-10, "eps"],
    "HammingCorr, L = 128": ["eps", "eps", "eps", "eps", "", "eps", "eps", "eps", "eps", "eps", "eps", "eps", 2.2e-16, "eps"],
    "HammingIndep, L = 16": ["eps", 1.8e-9, "eps", 3.9e-14, 5.4e-7, "eps", 6.4e-11, 4.4e-16, 1.9e-9, 7.2e-7, "eps", 1.1e-7, 2.3e-9, "eps"],
    "HammingIndep, L = 32": [1.4e-13, "", "eps", 2.5e-11, "", 4.3e-9, 2.3e-7, 2.4e-6, "", 4.9e-4, 1.7e-12, 4.9e-4, 2.3e-9, 3.1e-15],
    "HammingIndep, L = 64": [3.1e-8, "", 1.6e-12, 2.4e-10, "", 1.1e-5, "", "", 1.1e-9, "", 3.5e-5, "", "", 2.2e-6],
    "AutoCor": ["1-eps1", "", "1-eps1", 2.7e-11, "", 1 - 1.1e-7, "", 1 - 1.4e-5, 1 - 1.7e-10, 1 - 5.7e-7, 1 - 1.0e-6, "", 1 - 2.8e-15, ""],
    "Run of bits": ["eps", "eps", "eps", "eps", 1.1e-16, "eps", "eps", 6.5e-4, 4.8e-4, 3.4e-5, "eps", "eps", 7.7e-11, ""],
    "RandomWalk1 H": ["eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps", "eps"],
    "RandomWalk1 M": ["", "", 8.9e-16, "", "", "", "", "", "", "", "", "", "", "eps"],
    "RandomWalk1 R": [4.9e-5, "", 4.8e-4, "", "", "", 1.6e-4, "", "", "", "", "", "", ""],
    "RandomWalk1 J": ["", 6.9e-4, "", 7.6e-7, "", "", 3.0e-5, 2.6e-4, "", 9.8e-4, 6.9e-5, "", "", 2.0e-5],
    "RandomWalk1 C": ["", "", 5.1e-4, 4.3e-13, "", "", 4.0e-4, "", "", "", "", "", "", ""],
    "sts-frequency_within_block" :[1.2631584322520754e-209, 2.1763021976808273e-198, 5.130419580532256e-214, 1.3045576238292606e-175, 2.9779780687249056e-141, 9.005847440853053e-295, 2.5425478635472244e-114, 7.636213046069291e-186, 8.017448565929047e-127, 6.020505962779288e-149, 9.659027709716744e-59, 3.6723997745830484e-226,5.05445708412163e-221, 0.0]
}

# Convert the combined dictionary into a pandas DataFrame
df = pd.DataFrame(data_combined)
print(df)
#for col in df.columns:
#    if df[col].eq('').any(): 
#        df = df.drop(col, axis=1)
df = df.replace({'': 1, 'eps': 1.0e-300,'1-eps1':0.9999})
threshold = len(df) *0.2

# Drop columns with more than half values that are 1
df = df.drop(columns=df.columns[(df == 1).sum() > threshold])

#df = df.drop(columns=df.columns[(df == 1).any()])
from scipy.stats import ks_2samp
from itertools import combinations
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)# Assuming df is your DataFrame
print(df)
# Convert all columns to numeric type
df = df.drop(columns=["File Name"])
df = df.apply(pd.to_numeric, errors='coerce')
# Identify constant columns
#constant_columns = df.columns[df.nunique() == 1]

# Drop constant columns
#df = df.drop(columns=constant_columns)

# Convert remaining columns to numeric type
df = df.apply(pd.to_numeric, errors='coerce')

print(df)
# Get column names for pairwise comparison
columns = df.columns
print(columns)

# Create an empty DataFrame to store the results
ks_results = pd.DataFrame(columns=['Column 1', 'Column 2', 'KS statistic', 'p-value'])

# Perform pairwise KS test
from scipy.stats import ks_2samp
from tabulate import tabulate

# Initialize an empty list to store the results
results = []

# Iterate over combinations of columns
for col1, col2 in combinations(columns, 2):
    # Extract non-null values for the columns being compared
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()
    
    # Check if both datasets are not empty
    if not data1.empty and not data2.empty:
        # Perform KS test
        ks_statistic, p_value = ks_2samp(data1, data2)
        
        # Append the results to the list
        results.append([col1, col2, ks_statistic, p_value])

# Convert the list of results to a DataFrame
ks_results = pd.DataFrame(results, columns=['Column 1', 'Column 2', 'KS statistic', 'p-value'])

# Output the results as a table using the tabulate library
table = tabulate(ks_results, headers='keys', tablefmt='pipe')

# Print the table
print(table)
# Filter the DataFrame to keep only rows with p-values less than or equal to 0.005
ks_results_filtered = ks_results[ks_results['p-value'] <= 0.05]

# Output the filtered results as a table using the tabulate library
table_filtered = tabulate(ks_results_filtered, headers='keys', tablefmt='pipe')

# Print the filtered table
print("Filtered Results (p-values <= 0.05):\n")
print(table_filtered)
# Create a pivot table for better visualization
ks_pivot = ks_results.pivot(index='Column 1', columns='Column 2', values='KS statistic')
ks_pivot_f = ks_results_filtered.pivot(index='Column 1', columns='Column 2', values='KS statistic')
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(ks_pivot, annot=True, cmap='coolwarm', cbar=True, vmin=0, vmax=1)
plt.title('Pairwise KS Test p-values')
plt.xlabel('Column 2')
plt.ylabel('Column 1')
plt.tight_layout()

# Save the heatmap as an image file
plt.savefig('ks_test_heatmap.png')

# Show the heatmap
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(ks_pivot_f, annot=True, cmap='RdBu', cbar=True, vmin=0, vmax=1)
plt.title('Pairwise KS Test p-values <= 0.05')
plt.xlabel('Column 2')
plt.ylabel('Column 1')
plt.tight_layout()

plt.savefig('pairwiseksfilter.png')
plt.show()

from scipy.stats import pearsonr

# Create an empty list to store the results
results = []

# Perform pairwise Pearson correlation coefficient test

for col1, col2 in combinations(df.columns, 2):
    correlation, p_value = pearsonr(df[col1], df[col2])
    results.append([col1, col2, correlation, p_value])
# Write results in a table format
table_headers = ["Column 1", "Column 2", "Pearson Correlation", "p-value"]
table = tabulate(results, headers=table_headers, tablefmt="pretty")

# Print the table
print("Pairwise Pearson correlation coefficient test:")
print(table)

# Create a DataFrame from the results
pearson_results_df = pd.DataFrame(results, columns=["Column 1", "Column 2", "Pearson Correlation", "p-value"])

# Create a pivot table for better visualization
pearson_pivot = pearson_results_df.pivot(index="Column 1", columns="Column 2", values="Pearson Correlation")

pearson_results_df_filter = pearson_results_df[ks_results['p-value'] <= 0.05]
pearson_pivot_f = pearson_results_df_filter.pivot(index='Column 1', columns='Column 2', values='Pearson Correlation')

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_pivot, annot=True, cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Pairwise Pearson Correlation Coefficients")
plt.xlabel("Column 2")
plt.ylabel("Column 1")
plt.tight_layout()

# Save the heatmap as an image file
plt.savefig("pearson_correlation_heatmap.png")

# Show the heatmap
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(pearson_pivot_f, annot=True, cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Pairwise Pearson Correlation Coefficients filtered")
plt.xlabel("Column 2")
plt.ylabel("Column 1")
plt.tight_layout()

# Save the heatmap as an image file
plt.savefig("pearson_correlation_heatmap_filter.png")

# Show the heatmap
plt.show()

import pandas as pd 
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations

# Create an empty list to store the results
nmi_results = []

# Perform pairwise normalized mutual information test
for col1, col2 in combinations(df.columns, 2):
    nmi_score = adjusted_mutual_info_score(df[col1], df[col2])
    nmi_results.append([col1, col2, nmi_score])

# Convert the list of results to a DataFrame
nmi_results_df = pd.DataFrame(nmi_results, columns=['Column 1', 'Column 2', 'Normalized Mutual Information Score'])

# Output the results as a table
print("Pairwise Normalized Mutual Information Test:")
print(nmi_results_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a heatmap of mutual information scores
plt.figure(figsize=(10, 8))
heatmap_data = nmi_results_df.pivot(index='Column 1', columns='Column 2', values='Normalized Mutual Information Score')
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title('Pairwise Mutual Information Scores')
plt.xlabel('Column 2')
plt.ylabel('Column 1')
plt.tight_layout()

# Save the heatmap as an image file
plt.savefig('normalized_mutual_information_heatmap.png')

# Show the heatmap
plt.show()
