import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the main dataset
data_path = 'Spinout_data_longitudinal.csv'
data = pd.read_csv(data_path)
print(data.head())

# Select columns for analysis (excluding non-numeric columns)
selected_columns = ['intel_ipr', 'intel_ownership', 'intel_capital', 'others_capital', 
                    'round_cvc', 'total_cvc', 'sic', 'intel_industry', 'founder_executive', 
                    'founders_exp', 'founders_seniority', 'founders_kn_spec', 'patents', 
                    'uspc_classes', 'claims', 'backward_citations', 'forward_citations', 
                    'human_capital', 'intel_specific_human_capital', 'social_capital', 
                    'intel_specific_social_capital', 'acquired', 'closed', 'founding_team', 'employees']

# Ensure all selected columns are numeric
data_selected = data[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values
data_selected.dropna(inplace=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_selected)

# Apply Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_scaled)

# Predict anomalies
predictions = model.predict(X_scaled)

# Perform PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the results using the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='viridis', marker='o')
plt.title('Isolation Forest - Anomaly Detection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
