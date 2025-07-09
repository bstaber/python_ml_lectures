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

## Pure Python example using the standard libraries

We first show off how to implement the linear regression problem using only lists and standard libraries.

We simulate a linear relationship with added Gaussian noise.

```python
import random

# Parameters
n = 100
a_true = 2.0
b_true = 1.0
noise_std = 1.0

# Generate toy data
X_train = [random.uniform(0, 10) for _ in range(n)]
y_train = [a_true * x + b_true + random.gauss(0, noise_std) for x in x_values]
```

```python {marimo}
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
```

We compute the slope and intercept using the mean, covariance, and variance. These functions are not built-in in Python, so we can implement them ourselves.

```python
def mean(values):
    return sum(values) / len(values)

def covariance(x, y, x_mean, y_mean):
    return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / len(x)

def variance(values, mean_value):
    return sum((v - mean_value) ** 2 for v in values) / len(values)

x_mean = mean(X_train)
y_mean = mean(y_train)

cov_xy = covariance(X_train, y_train, x_mean, y_mean)
var_x = variance(X_train, x_mean)

a = cov_xy / var_x
b = y_mean - a * x_mean
```

## NumPy implementation

Let's perform the same but by relying on NumPy instead.

We simulate the same data but using NumPy's module `np.random`.

```python
import numpy as np
np.random.seed(42)
n_samples = 50

true_a = 2.5
true_b = 1.0
x = np.random.uniform(0, 10, size=n_samples)
noise = np.random.normal(0, 2, size=n_samples)

y = true_a * x + true_b + noise
```

We can next estimate the slope and intercept as well.

```python
x_mean = np.mean(x)
y_mean = np.mean(y)

cov_xy = np.mean((x - x_mean) * (y - y_mean))
var_x = np.mean((x - x_mean) ** 2)

a = cov_xy / var_x
b = y_mean - a * x_mean
```

## Visualize the fitted line

Compute predictions over a test set and compare the fitted line with the true line.

```python
X_test = np.linspace(0, 10, 100)
y_pred = a * x_test + b

plt.scatter(X_train, y_train, label="Noisy data")
plt.plot(X_test, true_a * X_test + true_b, color="green", linestyle="--", label="True line")
plt.plot(X_test, y_pred, color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression fit")
plt.legend()
```

```python {marimo}
x_mean = np.mean(x)
y_mean = np.mean(y)

cov_xy = np.mean((x - x_mean) * (y - y_mean))
var_x = np.mean((x - x_mean) ** 2)

a = cov_xy / var_x
b = y_mean - a * x_mean

x_test = np.linspace(0, 10, 100)
y_pred = a * x_test + b

plt.scatter(x, y, label="Noisy data")
plt.plot(x_test, true_a * x_test + true_b, color="green", linestyle="--", label="True line")
plt.plot(x_test, y_pred, color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression fit")
plt.legend()
```

## Summary

- We generated toy data with a known linear model and added noise.
- We implemented linear regression in two ways: manually using lists, and using NumPy.
- In both cases, we derived the slope and intercept from the definitions of covariance and variance.
