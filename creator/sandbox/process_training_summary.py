import pickle
from TrainingSummary import TrainingSummary

summary = pickle.load(open("training_summary.pkl", "rb"))
print(summary[0].loss)
