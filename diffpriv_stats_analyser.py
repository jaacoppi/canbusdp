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
import numpy as np

epsilons = [0.1, 1.0, 2.0]

dp_means = {eps: [] for eps in epsilons}
dp_stds = {eps: [] for eps in epsilons}

# Run dp_stats x times for each epsilon
run_count = 100
for epsilon in epsilons:
    for _ in range(run_count):
        dp_mean, dp_std = dp_stats(numbers, quantile_5, quantile_95, epsilon)
        dp_means[epsilon].append(dp_mean)
        dp_stds[epsilon].append(dp_std)

# Print results
print(f"Analysed lines: {len(numbers)}")
print(f"Min Value: {min_value}")
print(f"Max Value: {max_value}")
print(f"5% Quantile: {quantile_5}")
print(f"95% Quantile: {quantile_95}")
print(f"Median: {median_value}")
print(f"Mean: {mean_value}")
print(f"Standard deviation: {std_dev}")

print(f"DP values after {run_count} runs")
for epsilon in epsilons:
    print(f"DP epsilon {epsilon}: min {np.min(dp_means[epsilon])}, mean {np.mean(dp_means[epsilon])}, max {np.max(dp_means[epsilon])}, mean of standard deviations: {np.mean(dp_stds[epsilon])}")


