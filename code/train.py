#coding: UTF-8
from __future__ import division 
import ast
import os
import sys
import types
import math
from math import exp, expm1
import matplotlib.pyplot as plt

import cv2
import numpy as np

number = 5*2500
visited = np.ones(number)

class WeakClassifier(object):
	def __init__(self, dots, feature_func, func_label, polarity, theta):
		self.dots = dots
		self.feature_func = feature_func
		self.func_label = func_label
		self.polarity = polarity
		self.theta = theta

	def __str__(self):
		return "dots: a:%d, b:%d, c:%d, d:%d" % (dots[0], dots[1], dots[2], dots[3])

	def eva(self, sub_img_ii):
		if self.polarity*self.feature_func(sub_img_ii) < self.polarity*self.theta:
			return 1
		else:
			return 0

class StrongClassifier(object):
	def __init__(self, window=19, step_factor=1.25, info=None):
		self.weakers = []
		self.weights = []
		self.cor = 0
		self.window = window
		self.step_factor = step_factor
		if info is not None:
			for i in range(info.shape[0]):
				a, b, c, d, label, po, theta, weight = info[i, :]
				a, b, c, d = int(a), int(b), int(c), int(d)
				func = generate_func([a, b, c, d], int(label))
				self.weakers.append(WeakClassifier([a, b, c, d], func, label, po, theta))
				self.weights.append(weight)
			self.cor = sum(self.weights)/2.

	def add_weaker(self, weaker, weight):
		self.weakers.append(weaker)
		self.weights.append(weight)
		#self.cor = sum(self.weights)/2.
		self.cor += (weight)/2.

	def save_info(self):
		array = np.empty((len(self.weakers), 8))
		for (i, weaker) in enumerate(self.weakers):
			array[i, :4] = weaker.dots
			array[i, 4] = weaker.func_label
			array[i, 5] = weaker.polarity
			array[i, 6] = weaker.theta
			array[i, 7] = self.weights[i]
		return array
		
	def eva(self, img):
		results = np.array([weaker.eva(img) for weaker in self.weakers])
		return sum(self.weights*results) >= self.cor

	def eva_img(self, img):
		window = self.window
		faces = []
		while window < img.shape[0]:
			for i in range(0, img.shape[0]-window, int(window/2)):
				for j in range(0, img.shape[1]-window, int(window/2)):
					sub_img = subsampling(self.window, img[i:i+window, j:j+window])
					if self.eva(get_ii(sub_img)):
						faces.append([i, j, window])
			window = int(window*self.step_factor)
		return faces

def subsampling(window, image):
	return cv2.resize(image, (window, window))

def get_ii(image):
	mat = image.copy()
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if i==0 and j==0:
				mat[i,j] = mat[i,j]
			elif i==0:
				mat[i,j] += mat[i,j-1]
			elif j==0:
				mat[i,j] += mat[i-1,j]
			else:
				mat[i,j] = mat[i,j] + mat[i-1,j] + mat[i,j-1] - mat[i-1,j-1]
	return mat

def rect(img_ii, a, b, c, d):
	try:
		if b-1 < 0 or a-1 <0: 
			ba = 0
		else:
			ba = img_ii[b-1, a-1]
		
		if a-1 <0: 
			da = 0
		else:
			da = img_ii[d,a-1]
		
		if b-1 < 0:
			bc = 0
		else:
			bc  = img_ii[b-1,c]
		
		dc = img_ii[d,c]
		return dc + ba - da - bc
 
	except:
		print "%d, %d, %d, %d" % (a, b, c, d)


def generate_func(dots, label):
	a, b, c, d = dots
	if label == 21:
		# two-rectangle A
		mid = int((d-b+1)/2)
		def feature_func(img_ii):
			white = rect(img_ii,a,b,c,mid + b - 1)
			black = rect(img_ii, a, mid + b, c, d)
			return white - black
	elif label == 12:
		# two-rectangle B
		mid = int((c-a+1)/2)
		def feature_func(img_ii):
			white = rect(img_ii, a, b, a + mid - 1, d)
			black = rect(img_ii, a + mid, b, c, d)
			return white - black
	elif label == 31:
		# three-rectangle A
		tra = int((d-b+1)/3)
		tra_above, tra_below = b+tra, b+2*tra
		def feature_func(img_ii):
			white = rect(img_ii, a, b, c, b + tra - 1)
			white += rect(img_ii, a, b + 2*tra, c, d)
			black = rect(img_ii, a, b + tra , c, b + 2*tra - 1)
			return white - black
	elif label == 13:
		# three-rectangle B
		tra = int((c-a+1)/3)
		tra_left, tra_right = a+tra, a+2*tra
		def feature_func(img_ii):
			white = rect(img_ii, a, b, a+tra-1, d)
			white += rect(img_ii, a+2*tra, b, c, d)
			black = rect(img_ii, a+tra, b, a+2*tra-1, d)
			return white - black
	elif label == 22:
		# four-rectangle
		midH = int((d-b+1)/2)
		midW = int((c-a+1)/2)
		def feature_func(img_ii):
			white = rect(img_ii, a, b, a+midW-1, b+midH-1)
			white += rect(img_ii, a+midW, b+midH, c, d)
			black = rect(img_ii, a+midW, b, c, b+midH-1)
			black += rect(img_ii,a , b+midH, a+midW-1,d )
			return white - black
	return feature_func

def weakerlist(dots,label,positive_iis,negative_iis, weakers):
	a, b, c, d = dots
	feature_func = generate_func([a, b, c, d], label)
	if feature_func is not None:
		f = feature_func
		g = types.FunctionType(f.func_code, f.func_globals, name = f.func_name, argdefs = f.func_defaults, closure = f.func_closure)		
		weaker = WeakClassifier((a,b,c,d),g, label, 9999999, 9999999)
		weakers.extend([weaker])

	return weakers

def create_weaker_pool(window, positive_iis, negative_iis, sampling):
	weakers = []

	# two-rectangle A	
	print "label 21 begin..."
	for a, b, c, d in sampling['21']:
		feature_func = generate_func([a, b, c, d], 21)
		if feature_func is not None:
			weakers = weakerlist([a,b,c,d],21,positive_iis, negative_iis,weakers)

	# two-rectangle B
	print "label 22 begin..."
	for a, b, c, d in sampling['12']:
		feature_func = generate_func([a, b, c, d], 12)
		if feature_func is not None:
			weakers = weakerlist([a,b,c,d],12,positive_iis, negative_iis,weakers)

	# three-rectangle A
	print "label 31 begin..."
	for a, b, c, d in sampling['31']:
		feature_func = generate_func([a, b, c, d], 31)
		if feature_func is not None:
			weakers = weakerlist([a,b,c,d],31,positive_iis, negative_iis,weakers)
			
	# three-rectangle B
	print "label 32 begin..."
	for a, b, c, d in sampling['13']:
		feature_func = generate_func([a, b, c, d], 13)
		if feature_func is not None:
			weakers = weakerlist([a,b,c,d],13,positive_iis, negative_iis,weakers)

	#four-rectangle
	print "label 4 begin..."
	for a, b, c, d in sampling['22']:
		feature_func = generate_func([a, b, c, d], 22)
		if feature_func is not None:
			#continue
			weakers = weakerlist([a,b,c,d],22,positive_iis, negative_iis,weakers)

	print "weakers completed..."

	return weakers


def learning(weakers, negative_imgs, weights_negative, positive_imgs, weights_positive, eIn):
	best_index, best_weaker, best_error = -1, None, 1

	Tplus  = weights_positive.sum()
	Tminus = weights_negative.sum()

	error   = np.ones(number)
	theta   = np.ones(number)
	polarity = np.ones(number)
	
	#indexlist = np.ones(number)
	
	error = error*99999


	for (key, classifier) in enumerate(weakers):
		#给定元素之前人脸样本权重和
		Splus = 0.
		#给定元素之前非人脸样本权重和
		Sminus = 0.
		matrix = []
		
		#dots = classifier.dots
		#indexlist[key] = key
		feature_func = classifier.feature_func
		#label = classifier.func_label
		#根据特征计算每个样本的特征值
		for (i,sub_img_ii) in enumerate(positive_imgs):
			goal = feature_func(sub_img_ii)
			matrix.append((goal,1,i))
		for (i,sub_img_ii) in enumerate(negative_imgs):
			goal = feature_func(sub_img_ii)
			matrix.append((goal,-1,i))
		#特征值从大到小排序
		matrix.sort(key=lambda x:x[0],reverse=True)
		
		# 是否包括当前的weight
		for item in range(len(matrix)):

			e1 = Sminus + (Tplus - Splus)
			e2 = Splus + (Tminus - Sminus)

			#tmp = min(e1,e2)
			if min(e1,e2) <= error[key] and min(e1,e2) > eIn:
				error[key]    = min(e1,e2)
				theta[key]    = matrix[item][0]
				#polarity[key] = matrix[item][1]
				#polarity[key] = matrix[item][1]

				if e1 < e2:
					polarity[key] = 1
				else:
					polarity[key] = -1

			#更新给定元素前的人脸以及非人脸的权重和
			if matrix[item][1]==1:
				Splus += weights_positive[matrix[item][2]]
			elif matrix[item][1]==-1:
				Sminus += weights_negative[matrix[item][2]]

		if key%1000 == 0:
			print("next:", key, weakers[key].dots, error[key])

	print "matrix finish..."
	min_error = 999

	#寻找最小error
	for key in range( len(error) ):
		if error[key] < min_error and visited[key]==1:
			min_error = error[key]
			ht = key
			dot= weakers[key].dots
			label= weakers[key].func_label
			g= weakers[key].feature_func
	
	visited[ht] = 0

	best_weaker = WeakClassifier(dot,g, label, polarity[ht], theta[ht])
	print("best:", best_weaker.dots, min_error)

	return best_weaker, min_error

def boosting(weakers, negative_imgs, positive_imgs, T, eIn):
	classifier = StrongClassifier()

	negative_imgs = np.array(negative_imgs)
	positive_imgs = np.array(positive_imgs)

	weights_negative = np.ones(negative_imgs.shape[0])
	weights_positive = np.ones(positive_imgs.shape[0])	
	
	weights_negative /= (2*weights_negative.sum())
	weights_positive /= (2*weights_positive.sum())

	# 初始化weight

	for t in range(T):

		weights_sum = weights_positive.sum() + weights_negative.sum()
		weights_negative /= weights_sum
		weights_positive /= weights_sum

		weaker, error_rate = learning(weakers, negative_imgs, weights_negative, positive_imgs, weights_positive, eIn)
		
		print(t, error_rate)
		beta = error_rate / (1-error_rate)

		#alpha = 0.5 * np.log( (1-error_rate)/max(error_rate,exp(1e-16)) )

		#print alpha

		classifier.add_weaker(weaker, np.log(1./beta))

		for (i, one) in enumerate(positive_imgs):
			weights_positive[i] *= weaker.eva(one) == 1 and beta or 1
			#weights_positive[i] *= weaker.eva(one) == 1 and exp(-alpha) or exp(alpha)

		for (i, one) in enumerate(negative_imgs):
			weights_negative[i] *= weaker.eva(one) == 1 and 1 or beta
			#weights_negative[i] *= weaker.eva(one) == 1 and exp(alpha) or exp(-alpha)
			
		

	return classifier


def train(negative_imgs, positive_imgs, window, T, sampling, eIn):
	negative_imgs = [subsampling(window, img)/255. for img in negative_imgs]
	positive_imgs = [subsampling(window, img)/255. for img in positive_imgs]
	
	negative_iis = [get_ii(img) for img in negative_imgs]
	positive_iis = [get_ii(img) for img in positive_imgs]

	weakers = create_weaker_pool(window, positive_iis, negative_iis, sampling)
	return boosting(weakers, negative_iis, positive_iis, T, eIn)

def test(classifier, negative_imgs, positive_imgs):
	t2t, t2f, f2t, f2f = 0, 0, 0, 0
	print("to interger images ...")
	negative_iis = [get_ii(img/255.) for img in negative_imgs]
	positive_iis = [get_ii(img/255.) for img in positive_imgs]

	for ii in positive_iis:
		if classifier.eva(ii):
			t2t += 1
		else:
			t2f += 1

	for ii in negative_iis:
		if classifier.eva(ii):
			f2t += 1
		else:
			f2f += 1

	return t2t, t2f, f2t, f2f

def write_faces(img, faces):
	faced = img.copy()
	for face in faces:
		i, j, window = face
		cv2.rectangle(faced, (i, j+window), (j, i+window), (0, 255, 0), 1)
		# pts = np.array([[i, j], [i+window, j], [i+window, j+window], [i, j+window]], np.int32)
		# pts = pts.reshape(-1, 1, 2)
		# faced = cv2.polylines(faced, [pts], True, (0, 0, 255))
	return faced
	
if __name__ == "__main__":
	if sys.argv[1] == "train":
		nega_dir = "train/non-face"
		posi_dir = "train/face"
		sIn, tIn, eIn, pIn, save_file = sys.argv[2:] 

		T = int(tIn)
		eIn = float(eIn)

		window = 19
		save_file = save_file


		g = open('sampling2500.txt', 'r')
		info  = g.read()
		sampling = ast.literal_eval(info)
		positive_imgs = [cv2.imread(os.path.join(posi_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(posi_dir)]
		negative_imgs = [cv2.imread(os.path.join(nega_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(nega_dir)]

		npos = len(positive_imgs)
		nneg = int(math.floor( npos*float(pIn)  ))
		negative_imgs = negative_imgs[1:nneg]


		negative_imgs = np.array(negative_imgs)
		positive_imgs = np.array(positive_imgs)

		weights_negative = np.ones(negative_imgs.shape[0])
		weights_positive = np.ones(positive_imgs.shape[0])	
	
		weights_negative /= (2*weights_negative.sum())
		weights_positive /= (2*weights_positive.sum())

		T = int(T)

		print "Begin: T =" + str(T) + " error_min =" + str(eIn) + " p =" + str(pIn) + " save as " + str(save_file) +"......"
		classifier = train(negative_imgs, positive_imgs,  window, T, sampling, eIn)
		np.save(save_file, classifier.save_info())

	elif sys.argv[1] == "test-mit":
		test_dir = "test/"
		classifier_file = "weaker25003030.npy"
		nega_dir = os.path.join(test_dir, "non-face")
		posi_dir = os.path.join(test_dir, "face")
		print("loading images ...")
		negative_imgs = [cv2.imread(os.path.join(nega_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(nega_dir)]
		negative_imgs = negative_imgs[1:496]
		positive_imgs = [cv2.imread(os.path.join(posi_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(posi_dir)]
		print("loading classifier ...")
		info = np.load(classifier_file)
		classifier = StrongClassifier(info=info)
		weakers = classifier.weakers
		weight = classifier.weights

		tmatrix = []
		fmatrix = []

		for i in range( len(weakers) ):
			count = 0
			tmp = []
			tmpweight = []
			for item in weakers:
				if count < i+1:
					tmp.append(item)
					tmpweight.append(weight[count])
					count += 1
				else:
					break
			classifier.weakers = tmp
			classifier.weights = tmpweight
			t2t, t2f, f2t, f2f = test(classifier, negative_imgs, positive_imgs)
			TP = t2t
			FP = f2t
			FN = t2f
			TN = f2f
			FPR = FP / (FP+TN)
			TPR = TP / (TP+FN)
			#print(t2t, t2f, f2t, f2f)
			print t2t/max(1,t2f+t2t),
			print t2t/max(1,f2t+t2t)
			fmatrix.append(FPR)
			tmatrix.append(TPR)
			#print t2t/max(1,t2f+t2t)
			#print t2t/max(1,f2t+t2t)
		plt.plot(fmatrix,tmatrix)
		#plt.plot(fmatrix)
		#plt.ylabel('some numbers')
		plt.show()
		# weakers = classifier.weakers
		# for item in weakers:
		# 	print item.polarity,
		# 	print item.theta,
		# 	print item.dots
		# #exit()
		# t2t, t2f, f2t, f2f = test(classifier, negative_imgs, positive_imgs)
		# print(t2t, t2f, f2t, f2f)
		# print t2t/max(1,t2f+t2t)
		# print t2t/max(1,f2t+t2t)
	elif sys.argv[1] == "main":
		n = 1
		classifier = StrongClassifier(info=np.load("weaker250030.npy")[:n])
		img = cv2.imread("/Users/nathan/documents/python/face/image.jpg", cv2.IMREAD_GRAYSCALE)
		faces = classifier.eva_img(img/255.)
		img = write_faces(img, faces)
		cv2.imwrite("res2218.jpg", img)
