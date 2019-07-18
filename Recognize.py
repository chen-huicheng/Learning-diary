#!/usr/bin/env python3

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image,ImageFilter
from tensorflow.examples.tutorials.mnist import input_data

class Recognize():

    def __init__(self):   #构造函数     将 Train 训练的模型导入

        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('Model/mnist_model.ckpt.meta')
        self.saver.restore(self.sess, 'Model/mnist_model.ckpt')

        self.graph = tf.get_default_graph()
        self.x_data = self.graph.get_tensor_by_name('x_data:0')
        self.y_data = self.graph.get_tensor_by_name('y_data:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')

        self.loss = self.graph.get_tensor_by_name('loss:0')
        self.train = self.graph.get_tensor_by_name('Variable/Adam:0')
        self.y = self.graph.get_tensor_by_name('y:0')


    def __del__(self):  #析构函数   讲学习后的模型保存 并 关闭回话
        print('#')
        self.saver.save(self.sess,'Model/mnist_model.ckpt')
        self.sess.close()
        print('#')


    def Picture(self,fileName):  # 通过文件路径 对图片处理 
        
        image = Image.open(fileName).convert('L')
        image = image.resize((28,28),Image.ANTIALIAS)
        image.save(fileName)

        tv = list(image.getdata())
        value = [(255-x)*1.0/255.0 for x in tv]
        #value = [(255-x)*1.0/255.0-0.4 for x in tv]
        #value = [x if(x>0) else 0 for x in value]
        v = np.array(value)
        v = v.reshape((28,28))
        a = np.max(v,1);
        b = np.max(v,0);
        for i in range(27,-1,-1):
            if a[i] == 0:
                v = np.delete(v,i,0)
            if b[i] == 0:
                v = np.delete(v,i,1)
        len0 = 28 - len(v)
        len1 = 28 - len(v[0])
        i0 = len0 // 2
        j0 = len0 - i0
        i1 = len1 // 2
        j1 = len1 - i1
        v = np.pad(v,((i0,j0),(i1,j1)),'constant',constant_values = (0,0))
        #self.printf(v)
        value = v.reshape(784)
        return value


    def recognizeNumber(self,image): # 将处理过得图片识别成 独热码

        result = self.sess.run(self.y,feed_dict={self.x_data:image,self.keep_prob:1.0})
        
        for x in result:
            for i in x:
                print('%.2f' % (i),end=' ')
            print()

        label = np.argmax(result, 1)
        return label
    
    def Binarization(self,path,savePath):
        image = cv2.imread(path,0)
        ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        im = Image.fromarray(image)
        im.save(savePath)


    def Learn(self,image,label):  # 通过提供 图片 和 label 训练模型
        self.sess.run(self.train,feed_dict={self.x_data:image,
            self.y_data:label,keep_prob:0.5})
        '''
        for x in image:
            self.printf(x,28,28)
        for x in label:
            print(x)
        '''


    def printf(self,data):  # 将图片以 0 1 形式输出
        
        for i in range(len(data)):
            for j in range(len(data[0])):
                #print("%.2f"%(data[i][j]),end=' ')
                if data[i][j] > 0.05:
                    print('1',end='')
                else:
                    print('0',end='')
            print()


if __name__=='__main__':         #用于测试：


    image = Image.open('image/1.jpg').convert('L')
    image.save('image/1.jpg')
    image = []
    re = Recognize()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_x = mnist.test.images[:20]
    test_y = mnist.test.labels[:20]
    label = re.recognizeNumber(test_x)
    print(label)
    print(np.argmax(test_y,1))

