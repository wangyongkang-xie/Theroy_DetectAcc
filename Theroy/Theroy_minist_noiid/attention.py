'''
by:wangyongkang
autoencoder
'''
import logging
import datetime
import os
import numpy as np
import tensorflow as tf
from Theroy_minist_noiid.detection import AutoEncoder

class Attention(object):
    
    def __init__(self, model = 'autoencoder', epochs=50):

        self.epochs = epochs
        # self.model = None
        # self.scaler_ = StandardScaler()
        self.model_path = os.path.join("ae_model.h5")
        if model == 'autoencoder':
            self.ae = AutoEncoder(epochs=self.epochs)

        if os.path.exists("attention.log"):
            os.remove('attention.log')

    def cal_weights(self, input_weights, num_of_features = 3000, y = -2):
        features = []
        for input_weight in input_weights:
            features.append(np.hstack(np.array([ item.ravel() for item in input_weight ])))         
        features = np.asarray(features)

        feature_selector = range(num_of_features)
        # if not os.path.isfile(self.model_path):
        #     self.ae.fit(feature_selector)

        # if len(features[0]) > num_of_features:
        #     features = np.array([ item[:num_of_features] for item in features ])

        feature_selected = []
        for idx in range(len(features)):
            feature_selected.append( features[idx][feature_selector] ) 
        feature_selected = np.asarray(feature_selected)

        with open('attention.log', 'a') as fw:
            fw.write( "{} [INFO] Calculating attention...\n".format(datetime.datetime.now()) )
            
            session = tf.Session()
            with session.as_default():
                with session.graph.as_default():
                    decision_scores_ = self.ae.detect(feature_selected)
                    weights = np.power( decision_scores_, y)
            
            # weights = np.array(weights) * len(features)
            
            fw.write( "{} [INFO] Calculating completed!\n".format(datetime.datetime.now()) )
            fw.write( '{} [Scores]:  {}\n'.format(datetime.datetime.now(), decision_scores_) )
            fw.write( '{} [Weights]: {}\n'.format(datetime.datetime.now(), weights) )
            fw.write( '------------------------\n')
                
        return weights
    
    def train_AE(self, feature_selector):
        print("[Begin Training AE] {}".format( datetime.datetime.now().strftime('%m-%d %H:%M:%S') ))

        data_dir = "../saved_weights_for_AE"
        files = [ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) ]
        weights = []
        for f in files:
            weights.append( np.load( os.path.join(data_dir, f), allow_pickle=True ) )
        features = []
        for input_weight in weights:
            features.append( np.hstack( np.array([ item.ravel() for item in input_weight ])) )
        features = np.asarray(features)
        print(features.shape)

        feature_selected = []
        for idx in range(len(features)):
            feature_selected.append( features[idx][feature_selector] ) 
        feature_selected = np.asarray(feature_selected)

        session = tf.Session()
        with session.as_default():
            with session.graph.as_default():
                self.model.fit(feature_selected)
                self.model.save(self.model_path)
        session.close()
        print("[End Training AE] {}".format( datetime.datetime.now().strftime('%m-%d %H:%M:%S') ) )

    def zero_out(self, weights, threshold = 0.8):
        
        print( "[INFO] Shape of the calculated attention", weights.shape )
        average = np.average(weights)
        print( "[INFO] Mean value of the calculated attention",  average, "SHAPE:", average.shape)
        print( "[INFO] threshold * average =", str(threshold * average) )
        for i in range(len(weights)):
            if weights[i] < ( threshold * average ):
                weights[i] = 0
        
        return weights

if __name__ == '__main__':
    # test_attention = Attention()
    # attention = test_attention.cal_weights(np.random.rand(10,3000))
    # # print(np.random.rand(10,3000).shape)
    # print(attention)
    input_weights = np.random.rand(4,6)
    # print(input_weights)
    features = []
    for input_weight in input_weights:
        features.append(np.hstack(np.array([item for item in input_weight])))
    features = np.asarray(features)
    feature_selector = range(4)
    print(features)
    feature_selected = []
    for idx in range(len(features)):
        feature_selected.append(features[idx][feature_selector])
    feature_selected = np.asarray(feature_selected)
    print(feature_selected)
    # attention[0] = 0.00001
    # attention[1] = 0.001
    # print( test_attention.zero_out(attention), test_attention.zero_out(attention).shape )