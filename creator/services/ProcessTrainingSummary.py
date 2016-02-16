import pickle
from TrainingSummary import TrainingSummary
from WordDistance import WordDistance

summary = pickle.load(open("training_summary.pkl", "rb"))

distance_calc = WordDistance()
score = distance_calc.sentenceScore(summary[0].sample)


#now we want to compute a bunch more statistics for each item in the list
print("Processing %s samples"%len(summary))


for s in summary:
	s = s.addStats()
	print(s.score)


#sum_stats = [summ.addStats() for summ in summary]

#pickle.dump(sum_stats, open("training_summary_add.pkl", "wb"))

#for s in sum_stats:
#	print(s.score)
