import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random

test_range = 5

#List test
l = []
for i in range(test_range):
    l.append([])
    for j in range(test_range):
        l[i].append(random.randint(1,10))
print(l,"\n")
print("Matrix generated.", "\n", "----------------------------")
print()

#NumPy ndarray datatype test
m = np.array(l)
print(m)
print()

#Datatype operations tests
print(m + m, "\n")
print(m - m, "\n")
print(m * m, "\n")
print(m / m, "\n")
print("Ndarray operations completed.", "\n", "----------------------------")
print()

#Matplotlib test
x = np.linspace(0, 10)
y = np.cos(x)

plt.plot(x, y)
plt.title('Hello Graph Test')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
print("Success.", "\n", "----------------------------")
print()

#Pandas test
p = pd.DataFrame(l)
print(p.head(), "\n")
print("DataFrame onboard.", "\n")
