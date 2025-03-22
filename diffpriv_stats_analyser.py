import numpy as np
import sys

from diffprivlib.tools import quantile, mean, std

import warnings




def dp_stats(numbers, lower_bound, upper_bound, epsilon):
    dp_mean = mean(numbers, epsilon=epsilon, bounds=(lower_bound, upper_bound))
    dp_std = std(numbers, epsilon=epsilon)
    return dp_mean, dp_std


def line_to_uint64(line):
    values = list(map(int, line.strip().split(";")))
    num = np.uint64(0)
    for v in values:
        num = (num << 8) | v  # Shift left and add next byte
    return num

# Read file and process
inputfile = sys.argv[1]
numbers = []
print(f"Reading {inputfile}")
with open(inputfile, "r") as file:
    for line in file:
        numbers.append(line_to_uint64(line))

# Convert to numpy array for stats
numbers = np.array(numbers, dtype=np.uint64)

# Compute statistics
print("Compiling statistics")
mean_value = np.mean(numbers)
median_value = np.median(numbers)
std_dev = np.std(numbers)
min_value = np.min(numbers)
max_value = np.max(numbers)
quantile_5 = np.percentile(numbers, 5)
quantile_95 = np.percentile(numbers, 95)

# print("Running DP calculations")
dp_mean_10, dp_std_10 = dp_stats(numbers, quantile_5, quantile_95, 1.0)
dp_mean_01, dp_std_01 = dp_stats(numbers, quantile_5, quantile_95, 0.1)
dp_mean_20, dp_std_20 = dp_stats(numbers, quantile_5, quantile_95, 2.0)


# Print results
print(f"Analysed lines: {len(numbers)}")
print(f"Min Value: {min_value}")
print(f"Max Value: {max_value}")
print(f"5% Quantile: {quantile_5}")
print(f"95% Quantile: {quantile_95}")
print(f"Median: {median_value}")
print(f"Mean: {mean_value}")
print(f"DP Mean (e 0.1): {dp_mean_01}")
print(f"DP Mean (e 1.0): {dp_mean_10}")
print(f"DP Mean (e 2.0): {dp_mean_20}")
print(f"Standard Deviation: {std_dev}")
print(f"DP Stantard deviation (e 0.1): {dp_std_01}")
print(f"DP Stantard deviation (e 1.0): {dp_std_10}")
print(f"DP Stantard deviation (e 2.0): {dp_std_20}")

