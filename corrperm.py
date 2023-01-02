import itertools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress

def import_data(filename):
  # Open the file
  with open(filename, 'r') as f:
    # Read the data from the file
    data = [float(x) for x in f.read().split()]

  # Return the data
  return np.array(data)

def plot_correlation(x, y, r, p, title, x_label, y_label):
  # Fit a linear line to the data
  slope, intercept = np.polyfit(x, y, 1)
  fit = intercept + slope * x

  line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

  # Create a figure and axes
  fig, ax = plt.subplots()

  # Plot the data and the fit
  ax.plot(x, y, linewidth=0, marker='s', label='Data points')
  ax.plot(x, intercept + slope * x, label=line)

  # Add a title, axis labels, and a legend
  ax.set_title(title)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.legend(facecolor='white')
  plt.show()

  return f"y={intercept:.2f}+{slope:.2f}x, r={r:.2f}"

def correlations(filenames):
  # Create an empty list to store the results
  results = []

  # Calculate all combinations of data series
  for x_filename, y_filename in itertools.combinations(filenames, 2):
    # Import the data from the files
    x = import_data(x_filename)
    y = import_data(y_filename)

    # Calculate Pearson's correlation coefficient and p-value
    r, p = pearsonr(x, y)
    pearson_fit = plot_correlation(x, y, r, p, 'Pearson', x_filename, y_filename)
    results.append((x_filename, y_filename, 'Pearson', r, p, pearson_fit))

    # Calculate Spearman's rank-order correlation coefficient and p-value
    r, p = spearmanr(x, y)
    spearman_fit = plot_correlation(x, y, r, p, 'Spearman', x_filename, y_filename)
    results.append((x_filename, y_filename, 'Spearman', r, p, spearman_fit))

    # Calculate Kendall's tau and p-value
    r, p = kendalltau(x, y)
    kendall_fit = plot_correlation(x, y, r, p, 'Kendall', x_filename, y_filename)
    results.append((x_filename, y_filename, 'Kendall', r, p, kendall_fit))

  # Return the results
  return results

# Example data
filenames = ['data1.txt', 'data2.txt', 'data3.txt']
results = correlations(filenames)

# Print the results
print('x,y,Method,r,p,line')
for result in results:
  x, y, method, r, p, line = result
  print(f'{x},{y},{method},{r:.3f},{p:.3f},{line}')
