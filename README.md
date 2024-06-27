# Anomaly Detection using Isolation Forest and PCA

This repository contains a Python implementation of anomaly detection using Isolation Forest and Principal Component Analysis (PCA). The project demonstrates how to apply these techniques on a dataset, visualize the results, and gain insights from the data.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Anomaly detection is a crucial aspect of data analysis, especially in identifying unusual patterns that do not conform to expected behavior. This project uses Isolation Forest for detecting anomalies and PCA for dimensionality reduction and visualization.

## Data

The data used in this project is a longitudinal dataset of spinout companies with various features such as intellectual property, capital, patents, and more. 

Sample data (Spinout data (longitudinal).csv):

```csv
spinout_id,year,founded,state,intel_ipr,intel_ownership,intel_capital,others_capital,round_cvc,total_cvc,sic,intel_industry,founder_executive,founders_exp,founders_seniority,founders_kn_spec,patents,uspc_classes,claims,backward_citations,forward_citations,human_capital,intel_specific_human_capital,social_capital,intel_specific_social_capital,acquired,closed,founding_team,employees
4,2003,2003,CA,0,0,0,0,0,0,7371,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0
4,2004,2003,CA,0,0,0,0,0,0,7371,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0
...


## Requirements
To run this project, you need the following libraries installed:
pandas
numpy
matplotlib
seaborn
scikit-learn
You can install these dependencies using the following command:
pip install pandas numpy matplotlib seaborn scikit-learn


## Usage
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
Place your dataset files (Spinout data (longitudinal).csv and AirEau_features_lag15.csv) in the project directory.

Run the anomaly_detection.py script:

bash
Copy code
python anomaly_detection.py


## Example Code
Here is a brief overview of the main code:

python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Load the main dataset
data_path = 'Spinout data (longitudinal).csv'
data = pd.read_csv(data_path)
print(data.head())

# Select columns for analysis
selected_columns = ['intel_ipr', 'intel_ownership', 'intel_capital', 'others_capital', 'round_cvc', 'total_cvc', 'sic', 'intel_industry', 'founder_executive', 'founders_exp', 'founders_seniority', 'founders_kn_spec', 'patents', 'uspc_classes', 'claims', 'backward_citations', 'forward_citations', 'human_capital', 'intel_specific_human_capital', 'social_capital', 'intel_specific_social_capital', 'acquired', 'closed', 'founding_team', 'employees']

# Ensure all selected columns are numeric
data_selected = data[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values
data_selected.dropna(inplace=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_selected)

# Perform PCA
n_components = 3
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Apply Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_scaled)

# Predict anomalies
predictions = model.predict(X_scaled)

# Create a DataFrame for the first few principal components
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Add predictions for color coding
pca_df['Anomaly'] = predictions

# Pair plot of the first few principal components
sns.pairplot(pca_df, vars=['PC1', 'PC2', 'PC3'], hue='Anomaly', palette='viridis')
plt.suptitle('Pair Plot of the First Three Principal Components')
plt.show()


## Results
The project visualizes anomalies in the dataset using Isolation Forest and PCA. Various plots, including pair plots and heatmaps, provide insights into the data and highlight the anomalous points.

## Contributing
Contributions are welcome! Please read the CONTRIBUTING.md file for guidelines on contributing to this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


### Additional Files
**CONTRIBUTING.md**

```markdown
# Contributing to Anomaly Detection Project

We welcome contributions to the Anomaly Detection project! Here are a few ways you can help:

1. **Reporting Bugs**: If you find a bug, please create an issue and provide as much information as possible.

2. **Feature Requests**: If you have an idea for a new feature, please create an issue and describe your idea in detail.

3. **Pull Requests**: If you want to contribute code, please fork the repository, create a new branch, and submit a pull request.

## Guidelines
- Ensure your code is well-documented.
- Write clear commit messages.
- Follow the existing coding style.

##LICENSE
MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.