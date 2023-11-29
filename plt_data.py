import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("training_log/cgan.csv")
x = [i for i in range(len(data))]

plt.plot(x, data["D_loss"], label="D_loss")
plt.plot(x, data["G_loss"], label="G_loss")
plt.ylabel("loss")
plt.xlabel("Step")
plt.legend()

plt.show()