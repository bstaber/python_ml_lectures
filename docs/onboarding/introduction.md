# Introduction

/// marimo-embed
    height: 800px
    mode: read
    app_width: wide

```python
@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create interactive sliders
    freq = mo.ui.slider(1, 10, value=2, label="Frequency")
    amp = mo.ui.slider(0.1, 2, value=1, label="Amplitude")

    mo.hstack([freq, amp])
    return

@app.cell
def __():
    # Plot the sine wave
    x = np.linspace(0, 10, 1000)
    y = amp.value * np.sin(freq.value * x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('Interactive Sine Wave')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.gca()
    return
```

///