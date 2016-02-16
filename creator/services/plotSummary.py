
import pickle
from TrainingSummary import TrainingSummary

new_s = pickle.load(open("training_summary_add.pkl", "rb"))
scores = [s.score for s in new_s]
loss = [s.loss for s in new_s]
time = [s.t for s in new_s]
smooth_loss = [s.smooth_loss for s in new_s]


import matplotlib.pyplot as plt

plt.plot(time, scores)
plt.show()
