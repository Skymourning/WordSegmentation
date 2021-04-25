# WordSegmentationNote: 
Please use Python 3.7 possibly with NumPy, other external packages are not allowed.
Exercise 1 (100 pts)
Description:
• There are generally 4 states in Chinese word segmentation: STATES=‘B’,‘I’,‘E’,‘S’. You are
required to apply the HMM to word segmentation on the given dataset.
• Totally, there are 5500 sentences for training and 1492 sentences for testing.
• In the training stage, you need to construct an initial state matrix, a transition matrix,
and an emission matrix by counting the frequncy of the state transitions on the training
data.
• In the testing stage, Firstly, you need to lable the hidden state sequence by using the
Viterbi algorithm. Secondly, output the result of word segmentation according to the
state sequence. Thirdly, compare the word segmentation with ground truth and calculate
the F1 score.
Grading Standard:
• train data, test data and label are given (train data and test data are given in train.txt,
test.txt and gold label can be otained from test_gold.txt)
– HMM model (e.g., the construction of the probability matrix and the Viterbi algorithm.)
– Evaluation process
• We will use the F1 score over all sequences in test set to evaluate your results.
• FILL IN the blank in HMM Class templete.
• Write a report to simply record your algorithm and results.
