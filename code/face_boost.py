#coding: UTF-8
from __future__ import division 
import ast
import os
import sys
import types
import math
from math import exp, expm1
import time
import cv2
import numpy as np

featureNum = 5*5000
isVistited = np.ones(featureNum)
#计算积分图
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

#计算矩形特征
def recFeature(img_ii, a, b, c, d):
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
#对于每个特征构建一个弱分类器
def createWeaker(dots,label,positive_iis,negative_iis, weakers):
	a, b, c, d = dots
	maxNum = sys.maxsize
	feature_func = create_Haar([a, b, c, d], label)
	if feature_func is not None:
		f = feature_func
		g = types.FunctionType(f.func_code, f.func_globals, name = f.func_name, argdefs = f.func_defaults, closure = f.func_closure)		
		weaker = WeakClassifier(dots,g, label, maxNum, maxNum)
		weakers.extend([weaker])

	return weakers

#构建弱分类器
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
#构建强分类器
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
				func = create_Haar([a, b, c, d], int(label))
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
		return facesimg动窗口设置
def imgsampling(window, image):
	return cv2.resize(image, (window, window))

def create_Haar(dots, label):
	a, b, c, d = dots
	if label == 21:
		# two-recFeatureangle A
		mid = int((d-b+1)/2)
		def feature_func(img_ii):
			white = recFeature(img_ii,a,b,c,mid + b - 1)
			black = recFeature(img_ii, a, mid + b, c, d)
			return white - black
	elif label == 12:
		# two-recFeatureangle B
		mid = int((c-a+1)/2)
		def feature_func(img_ii):
			white = recFeature(img_ii, a, b, a + mid - 1, d)
			black = recFeature(img_ii, a + mid, b, c, d)
			return white - black
	elif label == 31:
		# three-recFeatureangle A
		tra = int((d-b+1)/3)
		tra_above, tra_below = b+tra, b+2*tra
		def feature_func(img_ii):
			white = recFeature(img_ii, a, b, c, b + tra - 1)
			white += recFeature(img_ii, a, b + 2*tra, c, d)
			black = recFeature(img_ii, a, b + tra , c, b + 2*tra - 1)
			return white - black
	elif label == 13:
		# three-recFeatureangle B
		tra = int((c-a+1)/3)
		tra_left, tra_right = a+tra, a+2*tra
		def feature_func(img_ii):
			white = recFeature(img_ii, a, b, a+tra-1, d)
			white += recFeature(img_ii, a+2*tra, b, c, d)
			black = recFeature(img_ii, a+tra, b, a+2*tra-1, d)
			return white - black
	elif label == 22:
		# four-recFeatureangle
		midH = int((d-b+1)/2)
		midW = int((c-a+1)/2)
		def feature_func(img_ii):
			white = recFeature(img_ii, a, b, a+midW-1, b+midH-1)
			white += recFeature(img_ii, a+midW, b+midH, c, d)
			black = recFeature(img_ii, a+midW, b, c, b+midH-1)
			black += recFeature(img_ii,a , b+midH, a+midW-1,d )
			return white - black
	return feature_func

#对于5种不同 Haar特征构建不同的分类器
def allHaar_weaker(window, positive_iis, negative_iis, sampling):
	weakers = []
	# two-recFeatureangle A	
	for a, b, c, d in sampling['21']:
		feature_func = create_Haar([a, b, c, d], 21)
		if feature_func is not None:
			weakers = createWeaker([a,b,c,d],21,positive_iis, negative_iis,weakers)

	# two-recFeatureangle B
	for a, b, c, d in sampling['12']:
		feature_func = create_Haar([a, b, c, d], 12)
		if feature_func is not None:
			weakers = createWeaker([a,b,c,d],12,positive_iis, negative_iis,weakers)

	# three-recFeatureangle A
	for a, b, c, d in sampling['31']:
		feature_func = create_Haar([a, b, c, d], 31)
		if feature_func is not None:
			weakers = createWeaker([a,b,c,d],31,positive_iis, negative_iis,weakers)
			
	# three-recFeatureangle B
	for a, b, c, d in sampling['13']:
		feature_func = create_Haar([a, b, c, d], 13)
		if feature_func is not None:
			weakers = createWeaker([a,b,c,d],13,positive_iis, negative_iis,weakers)

	#four-recFeatureangle
	for a, b, c, d in sampling['22']:
		feature_func = create_Haar([a, b, c, d], 22)
		if feature_func is not None:
			#coniterationNumue
			weakers = createWeaker([a,b,c,d],22,positive_iis, negative_iis,weakers)

	return weakers

#从所有构建的分类器选择出最有的分类器
def learning(weakers, negative_imgs, weights_negative, positive_imgs, weights_positive):
	
	start_time = time.time()
	best_index, best_weaker, best_error = -1, None, 1
	#所有人脸样本权重和，即论文中的T+
	Tplus  = weights_positive.sum()
	#所有非人脸样本权重和，即论文中的T-
	Tminus = weights_negative.sum()
	error   = np.ones(featureNum)
	theta   = np.ones(featureNum)
	polarity = np.ones(featureNum)

	error = error*sys.maxsize

	for (key, classifier) in enumerate(weakers):
		#给定元素之前人脸样本权重和,即论文中S+
		Splus = 0.
		#给定元素之前非人脸样本权重和，即论文中S-
		Sminus = 0.
		matrix = []
		feature_func = classifier.feature_func
		#label = classifier.func_label
		#根据特征计算每个样本的特征值
		for (index,sub_img_ii) in enumerate(positive_imgs):
			goal = feature_func(sub_img_ii)
			matrix.append((goal,1,index))
		for (index,sub_img_ii) in enumerate(negative_imgs):
			goal = feature_func(sub_img_ii)
			matrix.append((goal,-1,index))
		#特征值从大到小排序
		matrix.sort(key=lambda x:x[0],reverse=True)
		#
		for item in range(len(matrix)):

			#计算最小误差
			e1 = Sminus + (Tplus - Splus)
			e2 = Splus + (Tminus - Sminus)
			if min(e1,e2) <= error[key]:
				error[key]    = min(e1,e2)
				theta[key]    = matrix[item][0]


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

	min_error = 999

	#寻找最小error
	for key in range( len(error) ):
		if error[key] < min_error and isVistited[key]==1:
			min_error = error[key]
			ht = key
			dot= weakers[key].dots
			label= weakers[key].func_label
			g= weakers[key].feature_func
	
	isVistited[ht] = 0

	best_weaker = WeakClassifier(dot,g, label, polarity[ht], theta[ht])
	print("best:", best_weaker.dots, min_error)

        end_time = time.time()

        print("learning overall time: ",end_time - start_time)

	return best_weaker, min_error

def save_face(img, faces):
	faced = img.copy()
	left = []
	right = []
	for face in faces:
		i, j, window = face
		if i>0 and j>0:
			left.append(i)
			right.append(j)
	left1 = min(left)
	left2 = max(left)
	right1 = min(right)
	right2 = max(right)
	cv2.recFeatureangle(faced, (left1,right1),(left2,right2),(0, 255, 0), 1)
	return faced

#提升算法获得强分类器
def AdaBoost(weakers, negative_imgs, positive_imgs, T):
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

		weaker, error_rate = learning(weakers, negative_imgs, weights_negative, positive_imgs, weights_positive)
		
		print(t, error_rate)
		beta = error_rate / (1-error_rate)

		classifier.add_weaker(weaker, np.log(1./beta))

		for (i, one) in enumerate(positive_imgs):
			weights_positive[i] *= weaker.eva(one) == 1 and beta or 1

		for (i, one) in enumerate(negative_imgs):
			weights_negative[i] *= weaker.eva(one) == 1 and 1 or beta
			
			if t%10 == 0:
				tmpString = "outName"+str(t)+".txt"
				outFile = open(tmpString,'w')
				np.save(outFile, classifier.save_info())
	return classifier

#训练
def train(negative_imgs, positive_imgs, window, T, sampling, eIn):
	negative_imgs = [subsampling(window, img)/255. for img in negative_imgs]
	positive_imgs = [subsampling(window, img)/255. for img in positive_imgs]
	
	negative_iis = [get_ii(img) for img in negative_imgs]
	positive_iis = [get_ii(img) for img in positive_imgs]

	weakers = allHaar_weaker(window, positive_iis, negative_iis, sampling)
	return AdaBoost(weakers, negative_iis, positive_iis, T, eIn)
#测试
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

if __name__ == "__main__":
	if sys.argv[1] == "train":
                train_start_time = time.time()
		nega_dir = "/media/hadoop/AARONHUANG/MIT Face Data/faces/face.train/train/non-face-totrain"
                
		posi_dir = "/media/hadoop/AARONHUANG/MIT Face Data/faces/face.train/train/face"
		iterationNum, save_file = sys.argv[2:] 

		T = int(iterationNum)

		window = 19
		save_file = save_file


		g = open('5000samples.txt', 'r')
		info  = g.read()
		sampling = ast.literal_eval(info)
		positive_imgs = [cv2.imread(os.path.join(posi_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(posi_dir)]
		negative_imgs = [cv2.imread(os.path.join(nega_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(nega_dir)]

		npos = len(positive_imgs)
		negative_imgs = negative_imgs


		negative_imgs = np.array(negative_imgs)
		positive_imgs = np.array(positive_imgs)

		weights_negative = np.ones(negative_imgs.shape[0])
		weights_positive = np.ones(positive_imgs.shape[0])	
	
		weights_negative /= (2*weights_negative.sum())
		weights_positive /= (2*weights_positive.sum())

		T = int(T)

		classifier = train(negative_imgs, positive_imgs,  window, T, sampling)
		np.save(save_file, classifier.save_info())
                train_end_time = time.time()
                print("training overall time:", train_end_time - train_start_time)

	elif sys.argv[1] == "test-mit":
		test_dir = "/media/hadoop/AARONHUANG/MIT Face Data/faces/face.test/test/"
		classifier_file = "weaker250030302.npy"
		nega_dir = os.path.join(test_dir, "non-face")
		posi_dir = os.path.join(test_dir, "face")
		negative_imgs = [cv2.imread(os.path.join(nega_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(nega_dir)]
                negative_imgs = negative_imgs[0:471]
		positive_imgs = [cv2.imread(os.path.join(posi_dir, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(posi_dir)]
		info = np.load(classifier_file)
		classifier = StrongClassifier(info=info)
		weakers = classifier.weakers
		weight = classifier.weights

		tmatrix = []
		fmatrix = []
                print(len(weakers))
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
			print(t2t, t2f, f2t, f2f)
			print t2t/max(1,t2f+t2t)
			print t2t/max(1,f2t+t2t)
			fmatrix.append(FPR)
			tmatrix.append(TPR)
	elif sys.argv[1] == "main":
		n = 30
		classifier = StrongClassifier(info=np.load("/media/hadoop/AARONHUANG/MIT Face Data/faces/code/result.npy")[:n])
		img = cv2.imread("/media/hadoop/AARONHUANG/MIT Face Data/faces/22.jpg", cv2.IMREAD_GRAYSCALE)
		faces = classifier.eva_img(img/255.)
		img = save_face(img, faces)
		cv2.imwrite("result.jpg", img)

