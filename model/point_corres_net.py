import tensorflow as tf
import numpy as np
import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_PATH,'../utils'))
import tf_util

class siamese:
    def __init__(self, is_training_pl, bn_dacay, batch_size, point_num):
        self.cloudpoints1 = tf.placeholder(tf.float32, shape = (batch_size,point_num,3))
        self.cloudpoints2 = tf.placeholder(tf.float32, shape = (batch_size,point_num,3))
        self.labels = tf.placeholder(tf.int32,shape = (batch_size,point_num,2))
        self.is_training_pl = is_training_pl
        with tf.variable_scope("siamese") as scope:
            self.features1 = self.network_f(self.cloudpoints1, self.is_training_pl)
            scope.reuse_variables()
            self.features2= self.network_f(self.cloudpoints2, self.is_training_pl)
        self.net = self.network_distance(self.features1, self.features2, self.is_training_pl) 
        self.loss = self.distance_loss()
        
    def network_f(self,x,is_training_pl):
        pass
        
        
        
    
    
    def network_distance(self,x1,x2,is_training_pl):
        pass
        
        
    def distance_loss(self):  
        pass
        
        
        
