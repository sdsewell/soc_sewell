import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# 1. Load the data
df = pd.read_csv('melissinos_table_7_5_simple.csv')

# Column 0 is Ring Index and column 1 is radial position (mm^2)
x = df.iloc[:, 1]
y = df.iloc[:, 0]

# 2. Perform Linear Regression
# slope: m, intercept: b
# stderr: uncertainty in slope, intercept_stderr: uncertainty in intercept
res = stats.linregress(x, y)

# 3. Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='royalblue', label='Data Points', alpha=0.7)

# Calculate the regression line: y = mx + b
line = res.slope * x + res.intercept
plt.plot(x, line, color='firebrick', label=f'Fit: y = {res.slope:.2f}x + {res.intercept:.2f}')

# 4. Add labels and legend
plt.title('Linear Regression Analysis', fontsize=14)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.grid(True, linestyle='--', alpha=0.6)

# Display results in a text box on the plot
stats_text = (f"Slope: {res.slope:.4f} ± {res.stderr:.4f}\n"
              f"Intercept: {res.intercept:.4f} ± {res.intercept_stderr:.4f}\n"
              f"R-squared: {res.rvalue**2:.4f}")

plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.legend()
plt.show()

# Print results to console
print(f"Slope: {res.slope} +/- {res.stderr}")
print(f"Intercept: {res.intercept} +/- {res.intercept_stderr}")