import csv
import numpy as np


def toFloat(data):
	s = data.shape
	new_data = np.zeros((s[0], s[1]))
	for i in range(s[0]):
		for j in range(s[1]):
			new_data[i, j] = float(data[i, j])
	return new_data


def toFloat_1D(data):
	s = data.shape
	new_data = np.zeros(s[0])
	for i in range(s[0]):
			new_data[i] = float(data[i])
	return new_data


def loadTrainData():
	l = []
	with open('../data/train.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line) #25001*386

	l.remove(l[0]) #25000*386
	l = np.array(l)
	train = l[:,0:385]
	ref = l[:,385]
	for i in range(len(train)):
		train[i][0] = '1'

	return toFloat(train), toFloat_1D(ref)


def loadTestData():
	l = []
	with open('../data/test.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line) #28001*385

	l.remove(l[0]) #28000*385
	l = np.array(l)
	test = l
	for i in range(len(test)):
		test[i][0] = '1'

	return toFloat(test)


def saveResult(result, file_name):
	with open(file_name, 'w', newline='') as fw:
		my_writer = csv.writer(fw)
		
		my_writer.writerow(['id', 'reference'])
		for i in range(len(result)):
			temp = []
			temp.append(i)
			temp.append(result[i])
			my_writer.writerow(temp)


def normalization(data):
	for i in range(len(data)):
		temp = np.linalg.norm(data[i])
		data[i] = data[i] / temp
	
	return data


def getresult(test, train, ref, k):
	result = np.zeros(len(ref))

	for i in range(len(test)):
		likelihood = np.dot(train, test[i])
		sum = 0

		#find k nearest neighbours
		for j in range(k):
			max_pos = np.argmax(likelihood, 0)
			sum = sum + ref[max_pos]
			likelihood[max_pos] = -1

		result[i] = sum / k
	
	return result



if __name__ == '__main__':

	k1 = 1
	k2 = 2

	#normalization
	train, ref = loadTrainData()
	train = normalization(train)
	test = loadTestData()
	test = normalization(test)

	#2 test
	test_resultk1 = getresult(test, train, ref, k1)
	test_resultk2 =  getresult(test, train, ref, k2)

	
	test_result = 0.6 * test_resultk1 + 0.4 * test_resultk2
	
	saveResult(test_result, 'submission.csv')



