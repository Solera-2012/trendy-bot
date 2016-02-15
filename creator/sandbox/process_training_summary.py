import pickle
from TrainingSummary import TrainingSummary
from WordDistance import WordDistance

summary = pickle.load(open("training_summary.pkl", "rb"))

distance_calc = WordDistance.WordDistance()
score = distance_calc.sentenceScore(summary[0].sample)
print(score)
