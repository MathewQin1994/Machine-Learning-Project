# train.py
# There are two main machine learning libraries we used to construct our models: Sklearn and Tensorflow.
#
# There are four objects: readData, classifier, CNN, ANN
# readData: read train data and change it into suitable format
# classifier:split train and test data, train SVM or LR model, then do cross validation
# CNN:split train and test data, train CNN model
# ANN:split train and test data, train ANN model
#
# And some functions:
# main(datakind,classfier):
# datakind determines kind data you want to use. datakind=’chinese’ or ‘num’
# classfier determines which model you want to use. Classfier=’SVM’ or ‘LR’ or ‘CNN’ or ‘ANN’
#
# get_feature(X,y):get Histogram of Oriented Gradient features
# include subfunction preprocess_hog(digits) and deskew(img)
#
# onehotlabel(label): transform label to one-hot-label for CNN and ANN

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib



SZ = 20          #训练图片长宽
MAX_WIDTH = 1000 #原始图片最大宽度
Min_Area = 2000  #车牌区域允许最大面积
PROVINCE_START = 1000
#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        # bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        # mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        bin_cells = bin[:5, :5], bin[5:10, :5], bin[10:15, :5], bin[15:, :5], \
                    bin[:5, 5:10], bin[5:10, 5:10], bin[10:15, 5:10], bin[15:, 5:10], \
                    bin[:5, 10:15], bin[5:10, 10:15], bin[10:15, 10:15], bin[15:, 10:15], \
                    bin[:5, 15:], bin[5:10, 15:], bin[10:15, 15:], bin[15:, 15:]
        mag_cells = mag[:5, :5], mag[5:10, :5], mag[10:15, :5], mag[15:, :5], \
                    mag[:5, 5:10], mag[5:10, 5:10], mag[10:15, 5:10], mag[15:, 5:10], \
                    mag[:5, 10:15], mag[5:10, 10:15], mag[10:15, 10:15], mag[15:, 10:15], \
                    mag[:5, 15:], mag[5:10, 15:], mag[10:15, 15:], mag[15:, 15:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def onehotlabel(label):#transform label to one-hot-label
    mlb = MultiLabelBinarizer()
    y=[tuple([i]) for i in label]
    y=mlb.fit_transform(y)
    return mlb,y


def print_best_score(gsearch, param_test):#output the best result of cross validation
    # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def get_feature(X,y):#get Histogram of Oriented Gradient features
    chars_train = list(map(deskew, X))
    chars_train = preprocess_hog(chars_train)
    # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
    chars_label = np.array(y)
    print('feature shape',chars_train.shape)
    return chars_train,chars_label

class readData(object):
    def __init__(self):
        pass
    def read_data_num(self):#read number data
        chars_train = []
        chars_label = []
        for root, dirs, files in os.walk("train\\chars2"):
            if len(os.path.basename(root)) > 1:
                continue
            root_int = ord(os.path.basename(root))
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                chars_train.append(digit_img)
                # chars_label.append(1)
                chars_label.append(root_int)

        # chars_train = list(map(deskew, chars_train))
        # chars_train = preprocess_hog(chars_train)
        # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
        chars_train = np.array(chars_train)
        chars_label = np.array(chars_label)
        print('num alph shape',chars_train.shape)
        return chars_train,chars_label

    def read_data_chinese(self):#read characters and chinese characters data
        chars_train = []
        chars_label = []
        for root, dirs, files in os.walk("train\\charsChinese"):
            if not os.path.basename(root).startswith("zh_"):
                continue
            pinyin = os.path.basename(root)
            index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                chars_train.append(digit_img)
                # chars_label.append(1)
                chars_label.append(index)
        chars_train=np.array(chars_train)
        print('chinese data shape',chars_train.shape)
        return chars_train,chars_label



class classifier(object):
    def __init__(self):
        pass

    def datasplit(self,X,y):#split train and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def SVM(self):#construct SVM model and do cross validation
        estimator=SVC()
        parameters={
            "C":range(1,10,2),
            "gamma":[0.5,0.8],
        }
        self.gsearch = GridSearchCV(estimator, param_grid=parameters, cv=5,return_train_score=True)
        self.gsearch.fit(self.X_train,self.y_train)
        gridscores=self.gsearch.cv_results_
        # print(gridscores)
        print_best_score(self.gsearch, parameters)
        score=self.gsearch.score(self.X_test,self.y_test)
        print("testscore:%.3f"%score)
        return gridscores


    def LR(self):#construct LR model and do cross validation
        estimator=LogisticRegression()
        parameters={
            "C":range(1,24,5),
            "penalty":['l1','l2'],
            # "multi_class ":['ovr','multinomial']
        }
        self.gsearch = GridSearchCV(estimator, param_grid=parameters, cv=5,return_train_score=True)
        self.gsearch.fit(self.X_train,self.y_train)
        gridscores=self.gsearch.cv_results_
        # print(gridscores)
        print_best_score(self.gsearch, parameters)
        score=self.gsearch.score(self.X_test,self.y_test)
        print("testscore:%.3f"%score)
        return gridscores


class CNN(object):#construct CNN model
    def __init__(self):
        pass

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def datasplit(self,X,y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def train(self,isTrain=False):
        x=tf.placeholder("float",shape=[None,20,20])
        y_=tf.placeholder("float",shape=[None,31])
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(x, [-1, 20, 20, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([5 * 5 * 64, 2048])
        b_fc1 = self.bias_variable([2048])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.weight_variable([2048, 31])
        b_fc2 = self.bias_variable([31])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        checkpoint_dir = 'CNNmodel'
        saver = tf.train.Saver()

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if isTrain:
            self.accuall=pd.Series([],name='train_accuracy')
            for i in range(3000):
                a=np.random.choice(self.X_train.shape[0],50)
                xbatch=self.X_train[a,:,:]
                ybatch=self.y_train[a,:]
                xbatch=xbatch/255
                # batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    # train_accuracy = accuracy.eval(session=sess,feed_dict={x:self.X_train[i:i+50,:,:], y_:self.y_train[i:i+50,:], keep_prob: 1.0})
                    train_accuracy = accuracy.eval(session=self.sess,feed_dict={x:xbatch, y_:ybatch, keep_prob: 1.0})
                    # train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    # a=x_image.eval(session=sess,feed_dict={x:xbatch})
                    saver.save(self.sess, 'CNNmodel/model.ckpt', global_step=i + 1)
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                    self.accuall[i]=train_accuracy
                # train_step.run(session=sess,feed_dict={x:self.X_train[i:i+50,:,:], y_:self.y_train[i:i+50,:], keep_prob: 0.5})
                train_step.run(session=self.sess,feed_dict={x:xbatch, y_:ybatch, keep_prob: 0.5})
                # train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            plt.figure(1)
            plt.plot(self.accuall,label='train_accuracy')
            plt.legend(loc='upper right')
            plt.show()
            print("test accuracy %g" % accuracy.eval(session=self.sess,feed_dict={x:self.X_test, y_:self.y_test, keep_prob: 1.0}))

        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                pass
            print(self.sess.run(W_conv1))

    def predict(self,X):
        x = tf.placeholder("float", shape=[None, 20, 20])
        # y_ = tf.placeholder("float", shape=[None, 31])
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(x, [-1, 20, 20, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([5 * 5 * 64, 2048])
        b_fc1 = self.bias_variable([2048])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.weight_variable([2048, 31])
        b_fc2 = self.bias_variable([31])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        checkpoint_dir = 'CNNmodel'
        saver = tf.train.Saver()
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            pass

        return self.sess.run(y_conv,feed_dict={x:X,keep_prob: 1.0})

class ANN(object):#construct ANN model
    def __init__(self):
        pass

    def run(self):
        length = len(self.X_test)
        x = tf.placeholder(tf.float32, [None, 256])
        W = tf.Variable(tf.zeros([256,31]))
        b = tf.Variable(tf.zeros([31]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder("float", [None, 31])
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        self.accuall=pd.Series([],name='train_accuracy')
        for i in range(10000):
            a = np.random.choice(self.X_train.shape[0], 50)
            xbatch = self.X_train[a, :]
            ybatch = self.y_train[a, :]
            if i % 100 == 0:
                # train_accuracy = accuracy.eval(session=sess,feed_dict={x:self.X_train[i:i+50,:,:], y_:self.y_train[i:i+50,:], keep_prob: 1.0})
                train_accuracy = accuracy.eval(session=self.sess, feed_dict={x: xbatch, y_: ybatch})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                self.accuall[i]=train_accuracy
            self.sess.run(train_step, feed_dict={x:xbatch,y_:ybatch})
        plt.figure(1)
        plt.plot(self.accuall, label='train_accuracy')
        plt.legend(loc='upper right')
        plt.show()
        print(self.sess.run(accuracy, feed_dict={x: self.X_test, y_: self.y_test}))

    def datasplit(self,X,y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)



def main(datakind,classfier):
    Dt=readData()
    if datakind=='num':
        X, y = Dt.read_data_num()
    elif datakind=='chinese':
        X, y = Dt.read_data_chinese()
    else:
        print('datakind invalid')
        return

    if classfier=='LR':
        CL = classifier()
        feature, label = get_feature(X, y)
        CL.datasplit(feature, label)
        gridscores = CL.LR()
        filename1='LRmodel/LRmodel'+datakind+'.m'
        joblib.dump(CL.gsearch.best_estimator_, filename1)
        result=pd.DataFrame(gridscores)
        filename2='LRmodel/CVresult'+datakind+'.xlsx'
        writer=pd.ExcelWriter(filename2)
        result.to_excel(writer,'CVresult')
        writer.save()
    elif classfier=='SVM':
        CL = classifier()
        feature, label = get_feature(X, y)
        CL.datasplit(feature, label)
        gridscores = CL.SVM()
        filename1='SVMmodel/SVMmodel'+datakind+'.m'
        joblib.dump(CL.gsearch.best_estimator_, filename1)
        result=pd.DataFrame(gridscores)
        filename2='SVMmodel/CVresult'+datakind+'.xlsx'
        writer=pd.ExcelWriter(filename2)
        result.to_excel(writer,'CVresult')
        writer.save()
    elif classfier=='ANN':
        ann=ANN()
        feature, label = get_feature(X, y)
        mlb, label = onehotlabel(label)
        ann.datasplit(feature, label)
        ann.run()
        writer=pd.ExcelWriter('ANNmodel/ANN.xlsx')
        ann.accuall.to_excel(writer,'ANN')
        writer.save()
    elif classfier=='CNN':
        cnn=CNN()
        mlb, y = onehotlabel(y)
        cnn.datasplit(X, y)
        cnn.train(isTrain=True)
        writer=pd.ExcelWriter('CNNmodel/CNN.xlsx')
        cnn.accuall.to_excel(writer,'ANN')
        writer.save()
    else:
        print('classfier invalid')
    return



if __name__=="__main__":
    main(datakind='chinese', classfier='CNN')

    # for i in ['chinese','num']:
    #     for j in ['LR','SVM']:
    #         main(datakind=i,classfier=j)
    # Dt=readData()
    # # # X, y = Dt.read_data_num()
    # X,y=Dt.read_data_chinese()
    # cnn = CNN()
    # mlb, y = onehotlabel(y)
    # cnn.datasplit(X, y)
    # cnn.train(isTrain=False)
    # pre=cnn.predict(cnn.X_test[0:3,:,:])

    # feature,label=get_feature(X,y)
    # CL=classifier()
    # CL.datasplit(feature,label)
    # gridscores=CL.LR()
    # joblib.dump(CL.gsearch.best_estimator_,'LRmodel.m')
    # mlb,y=onehotlabel(y)
    # cnn=CNN()
    # cnn.datasplit(X,y)
    # cnn.train(isTrain=True)
    # pre=cnn.predict(cnn.X_test[0:1,:,:])
    # ann=ANN()
    # ann.datasplit(feature,label)
    # ann.run()