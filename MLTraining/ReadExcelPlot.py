# Read csv without header
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/joseph/out.csv', header=None)
plt.plot(df)

plt.show()
