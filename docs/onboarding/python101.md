# Python 101

In this chapter, we will cover the fundamental building blocks of Python programming. These basics will prepare you for more advanced topics in numerical computing later on.

## Hello, World!

```python {marimo}
print("Hello, world!")
```

Let's start with a classic. Create a file `hello_world.py` and write:

```python
print("Hello, world!")
```

Run your first script using your favorite IDE and by simply typing in your terminal

```bash
$ python hello_world.py
```

## Variables and basic types

Python supports several basic data types. You can create variables and assign values to them like this:

```python
x = 42        # integer
pi = 3.14     # float
name = "Alice"  # string
is_valid = True  # boolean
```

You can check the type of a variable using the `type()` function:

```python
print(type(x))
print(type(pi))
print(type(name))
print(type(is_valid))
```

## Strings

Strings are sequences of characters. You can work with them in several ways:

```python
greeting = "Hello"
name = "Alice"
message = f"{greeting}, {name}!"  # f-strings
print(message)
```

Useful string methods include:

```python
text = "  Python Basics  "
text.strip()         # Remove whitespace
text.lower()         # Convert to lowercase
text.split()         # Split into a list of words
"-".join(["A", "B"]) # Join into a string
```

## Collections

### Lists

Lists are ordered collections of items:

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("kiwi")      # Add an item
print(fruits[0])           # Access by index
```

Useful methods include:

```python
fruits.insert(1, "mango")  # Insert at position
fruits.remove("banana")    # Remove first occurrence
fruits.pop()               # Remove and return last item
fruits.index("apple")      # Find index of item
fruits.count("apple")      # Count occurrences
fruits.sort()              # Sort in-place
fruits.reverse()           # Reverse the list
fruits.copy()              # Shallow copy
fruits.clear()             # Remove all items
```

List comprehensions offer a concise way to create lists:

```python
squares = [x**2 for x in range(5)]
```

is the same as doing

```python
squares = [0, 1, 4, 9, 16]
```

### Tuples

Tuples are similar to lists but cannot be changed (immutable):

```python
point = (2, 3)
x, y = point  # tuple unpacking
```

### Dictionaries

Dictionaries store key-value pairs:

```python
person = {"name": "Alice", "age": 30}
print(person["name"])    # Access value by key
```

Useful methods also include:

```python
person.keys()        # ['name', 'age']
person.values()      # ['Alice', 30]
person.items()       # dict_items([('name', 'Alice'), ('age', 30)]) -> This is more tricky, it returns an iterator
person.get("age")    # Safer access
person.get("height", "unknown")  # With default
person.update({"age": 31})       # Update or add key-value pairs
person.pop("age")    # Remove and return value
person.popitem()     # Remove and return a key-value pair
person.clear()       # Remove all items
```

### Sets

Sets store unique elements:

```python
unique_values = set([1, 2, 2, 3])
print(unique_values) # {1, 2, 3}
```

Useful methods include:

```python
unique_values.add(4)
unique_values.pop()               # Remove and return an arbitrary element
unique_values.remove(2)           # Remove element, error if not found
unique_values.discard(5)          # Remove element if exists, no error
unique_values.union({4, 5})       # Set union
unique_values.intersection({2, 3})# Set intersection
unique_values.difference({1})     # Set difference
unique_values.clear()             # Remove all elements
```

## Control flow

### Conditional statements

Use `if`, `elif`, and `else` to control what your program does based on conditions:

```python
if age > 18:
    print("Adult")
elif age == 18:
    print("Just turned adult")
else:
    print("Minor")
```

### Loops

#### For loops

```python
for fruit in fruits:
    print(fruit)
```

#### While loops

```python
i = 0
while i < 3:
    print(i)
    i += 1
```

#### Using `range`, `enumerate`, and `zip`

```python
for i in range(3):
    print(i)
```

```python
for i, fruit in enumerate(fruits):
    print(i, fruit)
```

```python
colors = ["red", "yellow", "purple"]
for fruit, color in zip(fruits, colors):
    print(f"{fruit} is {color}")
```

## Functions

Functions are blocks of code that can be reused:

```python
def greet(name):
    return f"Hello, {name}!"
```

You can define default values for parameters:

```python
def greet(name="world"):
    return f"Hello, {name}!"
```

## Importing modules

Modules allow you to use external code. Here's how to import some standard library modules.

The math module provides mathematical functions such as square roots, trigonometry, exponentials, and constants like π.
```python
import math

print(math.sqrt(16))      # Ssquare root
print(math.sin(math.pi))  # sine of pi
print(math.log(10))       # natural logarithm
```

You can also import only the functions you need:

```python
from math import sqrt, pi

print(sqrt(25))
print(pi)
```


The os module gives you access to operating system features such as file and directory handling.

```python
import os

print(os.getcwd())      # Get current working directory
print(os.listdir())     # List files in the current directory
```

Other useful examples:

```python
os.mkdir("test_folder")     # Create a new directory
os.rename("old.txt", "new.txt")  # Rename a file
```

Sometimes, it's helpful to give a shorter name to a module using as:

```python
import numpy as np  # common convention for NumPy
```

In the next chapter, we will explore numerical computing using NumPy and SciPy.
