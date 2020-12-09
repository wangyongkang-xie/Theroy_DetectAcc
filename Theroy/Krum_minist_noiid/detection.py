import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.losses import mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

class AutoEncoder():
  
    def __init__(self, hidden_neurons=None,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss=mean_squared_error, optimizer='adam',
                 epochs=50, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1):

        if hidden_neurons is None:
            hidden_neurons = [64, 32, 32, 64]
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size

        self.model = None

    def build_model(self, n_features):
        model = Sequential()
        

        # Input layer
        model.add(Dense(self.hidden_neurons[0], activation='relu', 
                            input_shape=(n_features, ), 
                            activity_regularizer=l2(self.l2_regularizer)))
        model.add(Dropout(self.dropout_rate))
    
        for i in range(1, len(self.hidden_neurons)):
            model.add(Dense( self.hidden_neurons[i], activation= self.hidden_activation, 
                                    activity_regularizer=l2(self.l2_regularizer)))
            model.add(Dropout(self.dropout_rate))

        # Output Layer
        model.add(Dense(n_features, activation=self.output_activation, 
                            activity_regularizer=l2(self.l2_regularizer)))
        
        model.compile(loss=mean_squared_error, optimizer='adam')
        
        print(model.summary())

        return model

    def fit(self, X):

        n_samples, n_features = X.shape[0], X.shape[1]

        self.scaler_ = StandardScaler()
        X_norm = self.scaler_.fit_transform(X)
        
        self.hidden_neurons.insert(0, n_features)

        X_copy = np.copy(X_norm)
        np.random.shuffle(X_norm)
        
        model = self.build_model(n_features)
        history_ = model.fit(X_norm, X_norm, epochs=self.epochs, batch_size=self.batch_size, 
                            shuffle=True, validation_split=self.validation_size,verbose=1).history
        self.hidden_neurons.pop(0)
        
        self.model = model
        model_path = os.path.join("ae_model.h5")
        model.save(model_path)

    def detect(self, input):
        input_copy = np.copy(input)

        self.scaler_ = StandardScaler()
        X_norm = self.scaler_.fit_transform(input_copy)

        model_path = os.path.join("ae_model.h5")
        loaded_model = load_model(model_path)

        pred = loaded_model.predict(X_norm)

        euclidean_sq = np.square(pred - self.scaler_.fit_transform(input_copy))
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

# if __name__ == "__main__":
    # ae = AutoEncoder()
    # # input = np.array([ np.zeros(64) for _ in range(64) ])
    # # for i in range(len(input)):
    # #     input[i][i] = 1
    # # ae.fit(input)
    # # print(ae.detect(input))
    # data_dir = "../saved_weights_for_AE"
    # num_of_features = 3000
    # feature_selector = range(num_of_features)
    # files = [ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) ]
    # weights = []
    # for f in files:
    #     weights.append( np.load( os.path.join(data_dir, f), allow_pickle=True ) )
    # features = []
    # for input_weight in weights:
    #     features.append( np.hstack( np.array([ item.ravel() for item in input_weight ])) )
    # features = np.asarray(features)
    # print(features.shape)
    #
    # feature_selected = []
    # for idx in range(len(features)):
    #     feature_selected.append( features[idx][feature_selector] )
    # feature_selected = np.asarray(feature_selected)
    #
    # # session = tf.Session()
    # # with session.as_default():
    #     # with session.graph.as_default():
    # ae.fit(feature_selected)
            # self.model.save(self.model_path)
    # session.close()
    # ae = AutoEncoder()
    # print(ae.build_model(784))