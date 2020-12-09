import numpy as np
import keras
import random
import time
import pickle
import codecs
from keras import optimizers
from attacks import same_value_attack, sign_flipping_attack, random_attack, backward
import datetime
import sys,os
import gc

class LocalModel(object):
    def __init__(self, model_config, data_collected):
        self.model_config = model_config
        self.model = model_config['model_json']
        #self.acc = {}#统计acc
        sgd = optimizers.SGD(lr=0.06)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= sgd,
              metrics=['accuracy'])
#feminist使用
        # self.x_train, self.y_train = data_collected[0]
        # self.x_test, self.y_test = data_collected[1]
        # self.x_valid, self.y_valid = data_collected[2]
        self.x_train, self.y_train = data_collected
    def get_weights(self):
        return self.model.get_weights()
        # returns a list of all weight tensors in the model, as Numpy arrays

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def train_one_round(self):

        print("Begin model compiling. {}".format(datetime.datetime.now().strftime('%m-%d %H:%M:%S')) )
        sgd = optimizers.SGD(lr=0.06)
        self.model.compile(optimizer = sgd,
                            loss = keras.losses.categorical_crossentropy,
                            metrics = ['accuracy'])

        # print("Begin running on test set. {}".format(datetime.datetime.now().strftime('%m-%d %H:%M:%S')) )
        # feminist使用
        # pre_score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # print('Pre-Test loss:', pre_score[0])
        # print('Pre-Test accuracy:', pre_score[1])
        print("local model fitting. {}".format(datetime.datetime.now().strftime('%m-%d %H:%M:%S')) )
        history = self.model.fit(self.x_train, self.y_train,
                  epochs=self.model_config['epoch_per_round'],
                  batch_size=self.model_config['batch_size'],
                  verbose=1)
                  # validation_data=(self.x_valid, self.y_valid))

        # score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        training_loss = np.mean(history.history['loss'])
        training_acc  = np.mean(history.history['acc'])
        print('Train loss:', training_loss )
        print('Train accuracy:', training_acc )
        return self.model.get_weights(), training_loss, training_acc
    # feminist使用
    # def validate(self):
    # 	score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
    # 	print('Validate loss:', score[0])
    # 	print('Validate accuracy:', score[1])
    # 	return score
    # feminist使用
    # def evaluate(self):
    # 	score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    # 	print('Test loss:', score[0])
    # 	print('Test accuracy:', score[1])
    # 	return score

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 6000

    # def __init__(self, datasource, gpu, attack_mode=0, writer=None):feminsit
    def __init__(self, gpu, attack_mode=0, writer=None,number=0):
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu
        self.acc ={}
        self.number = number
        self.local_model = None
        # self.datasource = datasource(writer, attack_mode) feminist使用
        self.writer = writer
        self.request_sid = None
        self.attack_mode = attack_mode
        self.label_assigned = -1

    ########## Socket Event Handler ##########
    def on_init(self, init_info):
        model_config = init_info
        self.request_sid = model_config['request_sid']

        print('------------------------------')
        # print('\n on init')
        print('preparing local data based on server model_config. {}'.format(datetime.datetime.now().strftime('%m-%d %H:%M:%S')) )
        # fake_data, my_class_distr = self.datasource.fake_non_iid_data() feminist使用
        self.local_model = LocalModel(model_config, self.writer)
        print("local model initialized done. {}".format(datetime.datetime.now().strftime('%m-%d %H:%M:%S')) )
        # ready to be dispatched for training

        client_ready_info = {
                            'train_size': self.local_model.x_train.shape[0],
                            # 'class_distr': my_class_distr  # for debugging, not needed in practice
                            }
        return client_ready_info

    def on_request_update(self, train_next_round_info):
        """

        """
        # req = args[0]
        req = train_next_round_info

        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
            print("req['weights_format'] == 'pickle'")
        else:
            weights = req['current_weights']

        print("Begin setting weights {}".format( datetime.datetime.now().strftime('%m-%d %H:%M:%S') ) )
        self.local_model.set_weights(weights)
        print("Begin local training {}".format( datetime.datetime.now().strftime('%m-%d %H:%M:%S') ) )
        my_weights, train_loss, train_accuracy= self.local_model.train_one_round()

        # for item in my_weights:
        #     print("shape:",item.ravel())
        if self.attack_mode == 1:    # same value
            my_weights = [ same_value_attack(item) for item in my_weights ]
        elif self.attack_mode == 2:  # sign flipping
            my_weights = [ sign_flipping_attack(item) for item in my_weights ]
        elif self.attack_mode == 3:  # random value
            my_weights = [ random_attack(item) for item in my_weights ]
        elif self.attack_mode == 4:  # under development
            my_weights = [ backward(my_weights[idx], weights[idx]) for idx in range(len(my_weights)) ]
        elif self.attack_mode == 5:  # wrong label
            pass

        # print("Clipping and adding noise")添加噪声
        # my_weights_1D = np.hstack(np.array([ item.ravel() for item in my_weights ]))
        # weights_1D = np.hstack(np.array([ item.ravel() for item in weights ]))
        # update_1D = (my_weights_1D - weights_1D)
        # if LA.norm(update_1D) > 4:
        #     my_weights = [ weights[idx] + ((my_weights[idx] - weights[idx]) / (LA.norm(update_1D)/4.0)) +  np.random.normal( scale=0.1, size=my_weights[idx].shape )
        #                         for idx in range(len(my_weights)) ]
        # else:
        #     my_weights = [ my_weights[idx] +  np.random.normal( scale=0.1, size=my_weights[idx].shape )
        #                         for idx in range(len(my_weights)) ]

        #
        # self.save_weights(str(self.number) + "_" + str(self.attack_mode) + "_" + str(self.label_assigned) + "_" + str(req['round_number']),
        #                     req['round_number'], my_weights)
        resp = {
            'round_number': req['round_number'],
            # 'weights': obj_to_pickle_string(my_weights),
            'weights': my_weights,
            'train_size': self.local_model.x_train.shape[0],
            # 'valid_size': self.local_model.x_valid.shape[0],feminist
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            # 'pre_train_loss': pre_train_loss,
            # 'pre_train_accuracy': pre_train_accuracy,
            'attack_mode': self.attack_mode,
            'assigned_label': self.label_assigned,
        }
        # if req['run_validation']:
        # 	valid_loss, valid_accuracy = self.local_model.validate()         feminist
        # 	resp['valid_loss'] = valid_loss
        # 	resp['valid_accuracy'] = valid_accuracy

        del self.local_model
        for _ in range(20):
            gc.collect()

        return resp

    def on_stop_and_eval(self, input):
        req = input
        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
            print("req['weights_format'] == 'pickle'")
        else:
            weights = req['current_weights']

        self.local_model.set_weights(weights)
        test_loss, test_accuracy = self.local_model.evaluate()
        resp = {
            'test_size': self.local_model.x_test.shape[0],
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        return resp

    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))

    def save_weights(self, writer, iteration, weights):
        print("Saving weights ...")

        path = os.path.join("../",'saved_weights', 'iteration_' + str(iteration))
        if not os.path.exists(path):
            os.makedirs(path)
        np.save( os.path.join(path, writer), weights)
        pass
        print("Weights saved")


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))



# if __name__ == "__main__":
# 	#print(sys.argv)
# 	if len(sys.argv) < 3:
# 		print("please input gpu core and attack mode")
# 	gpu = int(sys.argv[1])
#
# 	# 0 (default): normal, 1: same value attack, 2: sign_flipping_attack, 3: random_attack
# 	attack_mode = int(sys.argv[2])
# 	if len(sys.argv) == 4:
# 		label_assigned = int(sys.argv[3])
# 	else:
# 		label_assigned = -1
#
# 	print("client run on {}".format(gpu))
# 	#ktf.set_session(get_session())
# 	try:
# 		FederatedClient("127.0.0.1", 5000, datasource.Mnist, gpu, attack_mode, label_assigned)
# 	except ConnectionError:
# 		print('The server is down. Try again later.')
