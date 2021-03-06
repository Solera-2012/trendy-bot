import pickle
from TrainingSummary import TrainingSummary
from WordDistance import WordDistance

summary = pickle.load(open("training_summary.pkl", "rb"))

distance_calc = WordDistance()
score = distance_calc.sentenceScore(summary[0].sample)


#now we want to compute a bunch more statistics for each item in the list
print("Processing %s samples"%len(summary))

new_s = []
for s in summary:
	s = s.addStats()
	print(s.score)
	new_s.append(s)
	pickle.dump(new_s, open("training_summary_add.pkl", "wb"))

#sum_stats = [summ.addStats() for summ in summary]
