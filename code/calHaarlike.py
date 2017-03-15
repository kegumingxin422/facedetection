#coding: UTF-8
from __future__ import division 
import numpy as np
import math
import random
import ast
import types

def allHaar(file_name,window):
	option = [(2,1),(1,2),(3,1),(1,3),(2,2)]
	W = H = window
	dotsList = dict()
	count = 0
	for  item in option:
		h = item[0] 
		w = item[1]
		tmp = []
		for i in range(window):
			for j in range(window):
				for kH in range( int(math.floor( (H-i)/h ) )):
					#print int(math.floor((H-i)/h)),
					#print int(math.floor( (H-i)/h ) ),
					for kW in range( int(math.floor( (W-j)/w ) )):
						#print i+kH*h
						a = j
						b = i
						c = j+(kW+1)*w-1
						d = i+(kH+1)*h-1
						#print "(",
		 	 			#print a,b,c,d,
		 	 			#print ")",
		 	 			tmp.append( [a,b,c,d] )
		 	 			count += 1

		dotsList[str(h)+str(w)] = tmp
	print count

	exit()
	f = open (file_name, 'w')
	f.write(str(dotsList))
	f.close()
	print (count)
	print ("allHaar feature cal completed...")

def sampling(from_filename,desfile_name,cato):
	option = ['21','12','31','13','22']
	g = open(from_filename, 'r')
	info  = g.read()
	info = ast.literal_eval(info)
	dotsList = dict()
	for i in option:
		tmp = []
		index= random.sample(range(len(info[i])), cato)
		for item in index:
			tmp.append(info[i][item])
		dotsList[i] = tmp
	f = open (desfile_name, 'w')
	f.write(str(dotsList))
	f.close()
	print ("Sampling completed...")

if __name__ == "__main__":
	window = 19
	allHaar('allHaar.txt',window)
	#sampling('allHaar.txt','sampling3000.txt',3000)

