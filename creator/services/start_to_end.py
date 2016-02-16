import pickle
import matplotlib.pyplot as plt


from VanillaRNN import RNN
from TrainingSummary import TrainingSummary
from WordDistance import WordDistance

RNN = RNN('../input/case_sample.xml')
RNN.train(100000)

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

#sum_stats = [summ.addStats() for summ in summary]
pickle.dump(new_s, open("training_summary_add.pkl", "wb"))

scores = [s.score for s in new_s]
loss = [s.loss for s in new_s]
time = [s.t for s in new_s]

plt.plot(time, scores)
plt.show()
