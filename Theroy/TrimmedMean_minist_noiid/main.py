import pickle
import uuid
import torch
import random
import codecs
import numpy as np
import datetime
import argparse
import gc
import tensorflow as tf
from attention import Attention
import datasource
from models import GlobalModel, GlobalModel_MNIST_CNN, RESNET18_MODEL, LENET5_MODEL, LENET5_MODEL_FEMNIST
from fl_client import FederatedClient
from datasource import Mnist
class FLServer(object):

    MIN_NUM_WORKERS = 4
    # MIN_NUM_WORKERS = 10

    # MAX_NUM_ROUNDS = 100
    MAX_NUM_ROUNDS = 36
    NUM_CLIENTS_CONTACTED_PER_ROUND = 4
    # NUM_CLIENTS_CONTACTED_PER_ROUND = 10

    ROUNDS_BETWEEN_VALIDATIONS = 1
    # LENET5_MODEL_FEMNIST, "127.0.0.1", 5000, gpu, output=args.output, aggregation=args.aggregation
    def __init__(self, global_model,aggregation="normal_atten"):
      # FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000, gpu)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu
        self.global_model = global_model()
        self.ready_client_sids = set()

        # self.host = host
        # self.port = port
        self.client_resource = {}

        self.wait_time = 0

        self.model_id = str(uuid.uuid4())

        self.aggregation = aggregation
        self.attention_mechanism = Attention()
        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        #####

        self.invalid_tolerate = 0


    def handle_client_update(self, data):

        self.current_round_client_updates = data
        uploaded_weights = [ x['weights'] for x in self.current_round_client_updates ]

        if self.aggregation in ["normal_atten", "atten", "rule_out"]:

            if self.aggregation == "normal_atten":
                # Same atttention
                print("### Update with normal attention mechanism! ###")
                attention = np.tile( np.array([1.0]), len(uploaded_weights) )
            else:
                print("### Update with calculated attention mechanism! ###")
                # attention = self.attention_mechanism.cal_weights(np.array( uploaded_weights ))
                attention = self.attention_mechanism.cal_weights(np.array( uploaded_weights ))
                print("old attention", attention)
                # type(attention): <class 'numpy.ndarray'> shape (10, )

                if self.aggregation == "rule_out":
                    # Rule out
                    new_attention = np.zeros(attention.shape)
                    for idx in range(len(attention)):
                        if attention[idx] > np.mean(attention):
                            new_attention[idx] = 1.0
                    attention = new_attention
                    print("new attention", attention)


            attack_label = [ "{}_{}".format(x['attack_mode'], x['assigned_label'])
                                    for x in self.current_round_client_updates ]
            self.global_model.update_weights_with_attention(
                uploaded_weights,
                [x['train_size'] for x in self.current_round_client_updates],
                attention,
                attack_label
            )
            
        else:
            print("### Update with baseline methods! ###")
            self.global_model.update_weights_baseline(
                uploaded_weights,
                [x['train_size'] for x in self.current_round_client_updates],
                self.aggregation
            )

        aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
            [x['train_loss'] for x in self.current_round_client_updates],
            [x['train_accuracy'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates],
            self.current_round
        )
        if self.global_model.prev_train_loss is not None and self.global_model.prev_train_loss < aggr_train_loss:
            self.invalid_tolerate = self.invalid_tolerate + 1
        else:
            self.invalid_tolerate = 0
        self.global_model.prev_train_loss = aggr_train_loss

    def handle_client_eval(self, data):
        if self.eval_client_updates is None:
            return

        self.eval_client_updates = data

        # tolerate 30% unresponsive clients

        aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
            [x['test_loss'] for x in self.eval_client_updates],
            [x['test_accuracy'] for x in self.eval_client_updates],
            [x['test_size'] for x in self.eval_client_updates],
        )
        print("\n--------Aggregating test loss---------\n")
        print("aggr_test_loss", aggr_test_loss)
        print("aggr_test_accuracy", aggr_test_accuracy)
        print("best model at round ", self.global_model.best_round, ", get the best loss ", self.global_model.best_loss)
        print("== done ==")
        self.eval_client_updates = None  # special value, forbid evaling again


    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self, clients):

        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        print("\n ### Round ", self.current_round, "### \n")

        # print("request updates from", client_sids_selected)
        # by default each client cnn is in its own "room"

        # path = os.path.join("../",'saved_weights', 'iteration_' + str(self.current_round))
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # np.save( os.path.join(path, "server_weights"), self.global_model.current_weights)

        train_next_round_info = {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    # 'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'current_weights': self.global_model.current_weights,
                    'weights_format': 'not pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }
        return train_next_round_info

    def stop_and_eval(self):
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            #emit('stop_and_eval', {
            #		'model_id': self.model_id,
            #		'current_weights': obj_to_pickle_string(self.global_model.current_weights),
            #		'weights_format': 'pickle'
            #	}, room=rid)
            self.emit('stop_and_eval', {
                    'model_id': self.model_id,
                    # 'current_weights': obj_to_pickle_string(self.global_model.best_weight),
                    'current_weights': self.global_model.best_weight,
                    'weights_format': 'not pickle'
                }, room=rid)

    # def start(self):
    # 	self.socketio.run(self.app, host=self.host, port=self.port, log_output=False)

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered!')
def test(info,x_test, y_test,aggregation,train=True,iid=True):

    # device = torch.device("cpu")
    model_ = GlobalModel_MNIST_CNN()
    model = model_.build_model()
    model.set_weights(info['current_weights'])
    score = model.evaluate(x_test, y_test, verbose=1)
    if train:
        if iid:
            with open(aggregation + '_minist_iid_train_acc_结果存放.txt', 'a') as file_handle:
                file_handle.write(str(score[1]))
                file_handle.write('\n')
        else:
            with open(aggregation + '_minist_noniid_train_acc_结果存放.txt', 'a') as file_handle:
                file_handle.write(str(score[1]))
                file_handle.write('\n')
    else:
        if iid:
            with open(aggregation + '_minist_iid_test_acc_结果存放.txt', 'a') as file_handle:
                file_handle.write(str(score[1]))
                file_handle.write('\n')
        else:
            with open(aggregation + '_minist_noniid_test_acc_结果存放.txt', 'a') as file_handle:
                file_handle.write(str(score[1]))
                file_handle.write('\n')
    return score

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    # import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    make_print_to_file(path='./')
    Minst_1 = Mnist(20,10)
    writers_all = Minst_1.partitioned_by_rows(num_workers=10)
    # writers = writers_all["train"]#iid数据
    writers = Minst_1.client()#noniid数据
    # data = datasource.MINSIT_NONIID()
    train_x,train_y = Minst_1.global_train()[0],Minst_1.global_train()[1]
    data_test = Minst_1.test_data()
    x_test, y_test = data_test[0],data_test[1]
    # writers = data.client()
    # print(len(writers))
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 		  type=int, default = 0)
    parser.add_argument('--aggregation', 	type=str, 
                            choices = ["normal_atten", "atten", "rule_out", "TrimmedMean", "Krum", "GeoMed"],
                            default = "TrimmedMean")
    parser.add_argument('--output', 	  type=str, default = "stats.txt")
    parser.add_argument("--batch_size",   type=int, default = 100)
    parser.add_argument("--local_epoch",  type=int, default = 1)
    parser.add_argument("--select_ratio", type=float, default = 1)
    parser.add_argument("--attack_mode",  type=int, default = 3)
    parser.add_argument("--attack_ratio", type=float, default = 0.2)
    args = parser.parse_args()
    gpu = args.gpu

    # if gpu != -1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu
    #     K.tensorflow_backend._get_available_gpus()
    #
    #     config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU':10} )
    #     sess = tf.Session(config=config)
    #     keras.backend.set_session(sess)

    print("##### Arguements #####")
    print(args)
    print("##########")
    NUM_CLIENT = int( len(writers) * args.select_ratio )
    MAX_NUM_ROUNDS = 50
    server = FLServer(GlobalModel_MNIST_CNN, aggregation=args.aggregation)
    acc_avg = 0
    train_acc = {}
    test_loss = {}
    test_acc = {}
    for cur_round in range(MAX_NUM_ROUNDS):

        temp_session = tf.Session()
        # tf.compat.v1.disable_v2_behavior()
        # tf.compat.v1.disable_eager_execution()
        with temp_session.as_default():
            with temp_session.graph.as_default():

                ##### Prepare and Select Clients #####
                clients = []
                selected_writers = np.random.choice( len(writers), size=NUM_CLIENT, replace=False )
                
                attack_mode = [0] * len(selected_writers)
                attack_number = int( args.attack_ratio * len(selected_writers) ) 
                for attack_idx in range(attack_number):
                    attack_mode[attack_idx] = args.attack_mode
                print("Selected Writers", selected_writers, len(selected_writers))
                print("Attack Modes", attack_mode)

# feiminst数据集使用
#                 for selected_idx in range(len(selected_writers)):
#                     clients.append( FederatedClient("127.0.0.1", 5000, datasource.Mnist, gpu,
#                             attack_mode[selected_idx], writer=writers[ selected_writers[selected_idx] ]) )
                for selected_idx in range(len(selected_writers)):
                    clients.append(FederatedClient(gpu,\
                                                   attack_mode[selected_idx], writer=writers[ selected_writers[selected_idx] ], number=selected_writers[selected_idx]) )
                init_info = {
                                # 'model_json': server.global_model.model.to_json(),
                                'model_json': server.global_model.model,
                                'model_id': server.model_id,
                                'min_train_size': 500,
                                'data_split': (0.6, 0.3, 0.1), # train, test, valid
                                'epoch_per_round': args.local_epoch,
                                'batch_size': args.batch_size,
                                'request_sid': "rid", #request.sid,
                            }

                for client in clients:
                    client.on_init( init_info )
                ########

                train_next_round_info = server.train_next_round( clients )

                clients_resp = []
                for client_idx in range(len(clients)):
                    print( "Round {}/{} Client {}/{} is training TIME: {}".format(cur_round, MAX_NUM_ROUNDS, client_idx, NUM_CLIENT, datetime.datetime.now().strftime('%m-%d %H:%M:%S')) )
                    clients_resp.append( clients[client_idx].on_request_update(train_next_round_info) )
                    print( "-------------------------------------------" )
        temp_session.close()
        # for i in clients_resp:
        #     acc_avg =
        # acc_avg = sum(d['train_accuracy'] for d in clients_resp)/len(clients_resp)
        # if cur_round not in
        # train_acc[cur_round] = acc_avg
        # acc_avg = sum(clients_resp[i]['train_accuracy'])/len(clients_resp) for i in range(len(clients_resp))
        score_test = test(train_next_round_info, x_test, y_test,args.aggregation,train=False,iid=False)
        score_train = test(train_next_round_info, train_x,train_y, args.aggregation,train=True,iid=False)
        # test_loss[cur_round] = score[0]
        test_acc[cur_round] = score_test[1]
        train_acc[cur_round] = score_train[1]
        print('test_acc:',score_test[1])
        print('train_acc:',score_test[1])
        server.handle_client_update(clients_resp)
        while len(clients) != 0:
            del clients[0]
        del clients
        del clients_resp
        for _ in range(20):
            gc.collect()
    print('train_acc:',train_acc)
    print('test_loss:',test_loss)
    print('test_acc:',test_acc)


    # print(train_acc)


    # stop_and_eval_info =  {
    #                 'model_id': server.model_id,
    #                 # 'current_weights': obj_to_pickle_string(server.global_model.best_weight),
    #                 'current_weights': server.global_model.best_weight,
    #                 'weights_format': 'not_pickle'
    #             }
    # clients_stop_and_eval_resp = []
    # print( "\n-------------------Run Testing-------------------" )
    # ##### Prepare and Select Clients #####
    # clients = []
    # selected_writers = np.random.choice( len(writers), size=NUM_CLIENT, replace=False )
    #
    # attack_mode = [0] * len(selected_writers)
    # attack_number = int( args.attack_ratio * len(selected_writers) )
    # for attack_idx in range(attack_number):
    #     attack_mode[attack_idx] = args.attack_mode
    # print("Selected Writers", selected_writers, len(selected_writers))
    # print("Attack Modes", attack_mode)
    #
    # for selected_idx in range(len(selected_writers)):
    #     clients.append( FederatedClient("127.0.0.1", 5000, datasource.FEMNIST, gpu,
    #             attack_mode[selected_idx], writer=writers[ selected_writers[selected_idx] ]) )
    #
    # init_info = {
    #                 'model_json': server.global_model.model.to_json(),
    #                 'model_id': server.model_id,
    #                 'min_train_size': 500,
    #                 'data_split': (0.6, 0.3, 0.1), # train, test, valid
    #                 'epoch_per_round': args.local_epoch,
    #                 'batch_size': args.batch_size,
    #                 'request_sid': str(12313), #request.sid,
    #             }
    #
    # for client in clients:
    #     client.on_init( init_info )
    # ########
    # for client_idx in range(len(clients)):
    #     print( "Client {} is runnng testing".format(client_idx) )
    #     clients_stop_and_eval_resp.append( client.on_stop_and_eval(stop_and_eval_info) )
    #     print( "-------------------------------------------" )
    #
    #server.handle_client_eval(clients_stop_and_eval_resp)




    

