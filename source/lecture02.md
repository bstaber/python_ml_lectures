# NumPy basics

This chapter introduces NumPy, a powerful library for numerical computing in Python. You will learn how to create and manipulate arrays, perform vectorized operations, and explore some basic linear algebra tools.

## What is NumPy?

NumPy (Numerical Python) provides efficient data structures and operations for working with large numerical datasets. Compared to Python lists, NumPy arrays are more compact, faster, and support advanced operations such as broadcasting and linear algebra.

To use NumPy in your code, you need to import it:

```python
import numpy as np
```

## Creating arrays

You can create NumPy arrays from Python lists:

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
```

Useful functions for creating arrays include:

```python
np.zeros((2, 3))       # array of zeros
np.ones((3,))          # array of ones
np.full((2, 2), 7)     # filled with 7s
np.eye(3)              # identity matrix
np.arange(0, 10, 2)    # evenly spaced values
np.linspace(0, 1, 5)   # linearly spaced values
```

## Array properties

Each array has several useful properties:

```python
a.shape     # dimensions
a.ndim      # number of dimensions
a.size      # total number of elements
a.dtype     # data type
```

## Indexing and slicing

You can access and modify elements just like Python lists:

```python
x = np.array([[10, 20, 30], [40, 50, 60]])
x[1, 2]       # single element
x[:, 1]       # second column
x[0:2, 1:]    # submatrix
```

Boolean and fancy indexing:

```python
x[x > 30]         # boolean mask
x[[0, 1], [1, 2]] # access (0,1) and (1,2)
```

## Array operations

NumPy supports element-wise operations:

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b      # element-wise addition
a * b      # element-wise multiplication
a ** 2     # square each element
```

Functions that operate on entire arrays:

```python
np.sum(a)
np.mean(a)
np.max(a)
np.min(a)
np.std(a)
```

Broadcasting allows operations on arrays of different shapes:

```python
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
a + b
```

## Reshaping arrays

You can change the shape of an array:

```python
a = np.arange(6)
a.reshape((2, 3))
a.flatten()     # 1D copy
a.ravel()       # 1D view (if possible)
a.T             # transpose
```

Add new axes:

```python
a[:, np.newaxis]
np.expand_dims(a, axis=0)
```

## Linear algebra basics

NumPy includes basic linear algebra operations:

```python
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

np.dot(A, b)             # dot product
np.matmul(A, A)          # matrix multiplication
A @ A                    # same as matmul

np.linalg.inv(A)         # inverse
np.linalg.norm(A)        # norm
np.linalg.solve(A, b)    # solve Ax = b
```

## Random number generation

NumPy provides a random module for generating random numbers:

```python
np.random.seed(0)        # for reproducibility

np.random.rand(2, 3)     # uniform [0, 1)
np.random.randn(3)       # standard normal
np.random.integers(1, 10, size=(2, 2))  # random integers
np.random.choice([1, 2, 3], size=5)     # random choice
```

## Practical examples

### Compute Euclidean distance between points

```python
a = np.array([1, 2])
b = np.array([4, 6])
distance = np.linalg.norm(a - b)
```

### Normalize each row of a matrix

```python
X = np.array([[1, 2], [3, 4]])
row_norms = np.linalg.norm(X, axis=1, keepdims=True)
X_normalized = X / row_norms
```

### Monte Carlo estimate of pi

```python
np.random.seed(0)
N = 10000
x = np.random.rand(N)
y = np.random.rand(N)
inside = x**2 + y**2 <= 1
pi_estimate = 4 * np.sum(inside) / N
```
