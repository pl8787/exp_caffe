#coding=utf-8

import math
import numpy as np
import glob
import os
import random
import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
with open("sampleSubmission.csv", 'r') as sample_sub_file:
	line = sample_sub_file.readline()
	filds = line.strip().split(",")
	classes = filds[1:]

class_label_dict = {}
label = 0
for class_name in classes:
	class_label_dict[class_name] = label
	label += 1
print len(class_label_dict)

class_count_dict = {}
imgpath_dict = {}
for folder in directory_names:
    for fileNameDir in os.walk(folder):
        class_name = fileNameDir[0].split("\\")[-1]
        class_count_dict.setdefault(class_name, 0)
        imgpath_dict.setdefault(class_name, [])
        for fileName in fileNameDir[2]:
            class_count_dict[class_name] += 1
            imgpath_dict[class_name].append(class_name+os.sep+fileName)

print "Trainning set splitting..."
print "Spliting Train Set..."
train_set = set() #the absolute path of the image
test_set = set() # the absolute path of the image

train_file = open("kaggle_train_online.data", 'w')
test_file = open("kaggle_test_online.data", 'w')
for class_name in imgpath_dict:
    images = imgpath_dict[class_name]
    train_s = set(images)
    train_set |= train_s

FILE_PATH_PREFIX = "PATH_PREFIX\\" # not include the class_name

with open("sampleSubmission.csv", 'r') as sample_sub_file:
    line = sample_sub_file.readline()
    line = sample_sub_file.readline()
    while line:
        filds = line.strip().split(",")
        filename = filds[0]
        test_file.write("%s%s %s\n" %(FILE_PATH_PREFIX, filename, 0))
        line = sample_sub_file.readline()


print "Train size ", len(train_set)
#print "Test size ", len(test_set)
#print "Ratio ", len(test_set)*1.0/len(train_set)
for trainfile in train_set:
    train_file.write("%s%s %s\n" %(FILE_PATH_PREFIX, trainfile, class_label_dict[trainfile.split(os.sep)[0]]))
#for testfile in test_set:
#    test_file.write("%s%s %s\n" %(FILE_PATH_PREFIX, testfile, class_label_dict[testfile.split(os.sep)[0]]))
train_file.close()
test_file.close()
