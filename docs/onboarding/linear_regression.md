# Linear Regression

In this note, we implement linear regression in one dimension using two approaches:
1. A pure Python implementation using only lists and basic operations.
2. A NumPy implementation using the analytical solution.

We will also generate some synthetic data for demonstration.

## Mathematical formulation

We are given a dataset of $n$ input-output pairs $(x_i, y_i)_{i=1}^{n}$. We assume a linear relationship: $y_i = a x_i + b$. Our goal is to find the best slope $a$ and intercept $b$ that minimize the mean squared error:

$$ \mathrm{MSE}(a, b) = \frac{1}{n} \sum_{i=1}^n (y_i - (a x_i + b))^2 $$

The optimal parameters are given by:

$$
a = \frac{\sum (x_i - \overline{x})(y_i - \overline{y})}{\sum (x_i - \overline{x})^2}\,, \quad b = \overline{y} - a \overline{x}
$$

where $\overline{x}$ and $\overline{y}$ are the means of $x$ and $y$, respectively.


## Step 1: Generate toy data with noise

We simulate a linear relationship with added Gaussian noise:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 50

x = np.linspace(0, 10, n_samples)
true_a = 2.5
true_b = 1.0
noise = np.random.normal(0, 2, size=n_samples)

y = true_a * x + true_b + noise

plt.figure()
plt.scatter(x, y, label="Noisy data")
plt.plot(x, true_a * x + true_b, color="green", label="True line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Toy data with noise")
plt.legend()
plt.show()
```

We now use this data (converted to lists) for a manual implementation.

## Step 2: Pure Python implementation using lists

We compute the slope and intercept using the mean, covariance, and variance:

```python
x_list = x.tolist()
y_list = y.tolist()

def mean(values):
    return sum(values) / len(values)

def covariance(x, y, x_mean, y_mean):
    return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / len(x)

def variance(values, mean_value):
    return sum((v - mean_value) ** 2 for v in values) / len(values)

x_mean = mean(x_list)
y_mean = mean(y_list)

cov_xy = covariance(x_list, y_list, x_mean, y_mean)
var_x = variance(x_list, x_mean)

a = cov_xy / var_x
b = y_mean - a * x_mean

print(f"Estimated slope (pure Python): {a:.2f}")
print(f"Estimated intercept (pure Python): {b:.2f}")
```

## Step 3: NumPy implementation

We repeat the same process using NumPy, but without calling a solver:

```python
x_mean = np.mean(x)
y_mean = np.mean(y)

cov_xy = np.mean((x - x_mean) * (y - y_mean))
var_x = np.mean((x - x_mean) ** 2)

a = cov_xy / var_x
b = y_mean - a * x_mean

print(f"Estimated slope (NumPy): {a:.2f}")
print(f"Estimated intercept (NumPy): {b:.2f}")
```

## Step 4: Visualize the fitted line

```python
y_pred = a * x + b

plt.scatter(x, y, label="Noisy data")
plt.plot(x, true_a * x + true_b, color="green", linestyle="--", label="True line")
plt.plot(x, y_pred, color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression fit")
plt.legend()
plt.show()
```

## Summary

- We generated toy data with a known linear model and added noise.
- We implemented linear regression in two ways: manually using lists, and using NumPy.
- In both cases, we derived the slope and intercept from the definitions of covariance and variance.
