# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 22:51:59 2018

@author: fanat
"""



Article = input('Enter article: ')
Question = input('Enter question: ')
a_token =  word_tokenize(Article)
q_token =  word_tokenize(Question)
paragraph = np.zeros((60, 300, 300))
question = np.zeros((60, 30, 300))
empty_answer_idx = np.ndarray((60, 300))

word_counter = Counter()

for w in a_token:
	word_counter[w] += 1
for q in q_token:
	word_counter[q] += 1
    
    
w2v_300 = get_word2vec('Data/glove.840B.300d.txt', word_counter)

for j in range(len(a_token)):
    if j >= 300:
        break
    try:
        paragraph[0][j][:300] = w2v_300[a_token[j]]
    except KeyError:
        pass
    
for j in range(len(q_token)):
    if j >= 30:
        break
    try:
        question[0][j][:300] = w2v_300[q_token[j]]   
    except KeyError:
        pass

predictions_si, predictions_ei = sess.run([pred_si, pred_ei], feed_dict={
	input_tensors['p']:paragraph,
	input_tensors['q']:question,
	input_tensors['a_si']:empty_answer_idx,
	input_tensors['a_ei']:empty_answer_idx,
})
    
     
parag = a_token
f1 = []
pred_tokens = parag[int(predictions_si[i]):int(predictions_ei[i])+1]

print(' '.join(pred_tokens))

 