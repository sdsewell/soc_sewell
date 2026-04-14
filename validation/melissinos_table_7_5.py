import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Checking in: {script_dir}")
print(f"Files found: {os.listdir(script_dir)}")

# Build the full path to the CSV
csv_path = os.path.join(script_dir, 'melissinos_table_7_5.csv')

# Load the data
df = pd.read_csv(csv_path) 



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

# --- SET AXIS LIMITS HERE ---
plt.xlim(left=0)             # Starts X-axis at 0, lets the right side auto-scale
plt.ylim(0, 12)              # Sets Y-axis strictly from 0 to 12

# 4. Add labels and legend
plt.title('Melissinos Figure 7.28 From Table 7.5Linear Regression Analysis', fontsize=14)
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