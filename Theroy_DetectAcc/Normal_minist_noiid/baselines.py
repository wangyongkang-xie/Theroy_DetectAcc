# from math import log

import numpy as np
import hdmedians as hdm

from scipy.stats import trim_mean
class Baseline(object):
    def cal_Normal(self,input_weights):
        res = []

        for layer_idx in range( len(input_weights[0]) ):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [ item[layer_idx].flatten() for item in input_weights ]
            one_layer_set = np.array( one_layer_set )

            one_layer_set = np.mean(one_layer_set, axis=0)
            one_layer_set = np.reshape(one_layer_set, shape_cur_layer)

            res.append( one_layer_set )

        return res
    def cal_MarMed(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range( len(input_weights[0]) ):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [ item[layer_idx].flatten() for item in input_weights ]
            one_layer_set = np.array( one_layer_set )

            one_layer_set = np.median(one_layer_set, axis=0)
            one_layer_set = np.reshape(one_layer_set, shape_cur_layer)

            res.append( one_layer_set )

        return res

    def cal_TrimmedMean(self, input_weights, beta = 0.1):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range( len(input_weights[0]) ):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [ item[layer_idx].flatten() for item in input_weights ]
            one_layer_set = np.array( one_layer_set )
            print(one_layer_set)
            one_layer_results = trim_mean(one_layer_set, beta)
            print(one_layer_results)
            one_layer_results = np.reshape(one_layer_results, shape_cur_layer)

            res.append( one_layer_results )

        return res

    def cal_GeoMed(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range( len(input_weights[0]) ):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [ item[layer_idx].flatten() for item in input_weights ]
            one_layer_set = np.array( one_layer_set ).astype(float)

            one_layer_set = hdm.geomedian(one_layer_set, axis = 0)

            one_layer_set = np.reshape( np.array(one_layer_set), shape_cur_layer )
            res.append( one_layer_set )

        return res

    def cal_Medoid(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range( len(input_weights[0]) ):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [ item[layer_idx].flatten() for item in input_weights ]
            one_layer_set = np.array( one_layer_set ).astype(float)

            one_layer_set = hdm.medoid(one_layer_set, axis = 0)

            one_layer_set = np.reshape( np.array(one_layer_set), shape_cur_layer )
            res.append( one_layer_set )

        return res

    def cal_Krum(self, input_weights, num_machines = 10, num_byz = 2):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range( len(input_weights[0]) ):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [ item[layer_idx].flatten() for item in input_weights ]
            one_layer_set = np.array( one_layer_set ).astype(float)

            score = []
            num_near = num_machines - num_byz - 2
            for i, w_i in enumerate(one_layer_set):
                dist = []
                for j, w_j in enumerate(one_layer_set):
                    if i != j:
                        dist.append( np.linalg.norm(w_i - w_j) ** 2 )
                dist.sort(reverse=False)
                score.append(sum(dist[0:num_near]))
            i_star = score.index( min(score) )

            selected = one_layer_set[i_star]

            selected = np.reshape( np.array(selected), shape_cur_layer )
            res.append( selected )

        return res
    def cal_theroy(self,input_weights,num):
        """
        @param input_weights: 输入的各个client的权重
        @param num: 恶意攻击者个数
        """
        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer
            dic = {}
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            # print(one_layer_set)
            one_layer_set = np.array(one_layer_set).astype(float)
            for i in range(one_layer_set.shape[1]):
                lis = [[] for _ in range(num)]
                min = one_layer_set[:,i].min()
                max = one_layer_set[:,i].max()
                dis = (max - min)/num
                if dis == 0:
                    break
                for j, w_j in enumerate(one_layer_set[:,i]):
                    k = (w_j - min)//dis
                    k = int(k)
                    if k == num:
                        lis[k-1].append(j)
                    else:
                        lis[k].append(j)
                for m in range(one_layer_set.shape[0]):
                    if m not in dic.keys():
                        dic[m]=0
                    entropy = 0
                    for n in range(len(lis)):
                        if m in lis[n]:
                            tem = lis[n][:]
                            tem.remove(m)
                            p = len(tem) / (one_layer_set.shape[0] - 1)
                            if p == 0:
                                continue
                            else:
                                entropy += -p * np.log(p)
                        else:
                            p = len(lis[n]) / (one_layer_set.shape[0] - 1)
                            if p == 0:
                                continue
                            else:
                                entropy += -p * np.log(p)
                    dic[m] += entropy
            # sorted(dic.items(), key=lambda item: item[1])
            print(dic)
            need_del = []
            mean = sum(item[1] for item in dic.items())/len(dic)
            for item in dic.items():
                if item[1] <= mean:
                    need_del.append(item[0])
            print(need_del)
            one_layer_set = np.delete(one_layer_set,need_del,axis=0)
            selected = one_layer_set.mean(axis=0)
            selected = np.reshape(np.array(selected), shape_cur_layer)
            res.append(selected)

        return res


class Baseline_single(object):
    def cal_MarMed(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            one_layer_set = np.array(one_layer_set)

            one_layer_set = np.median(one_layer_set, axis=0)
            one_layer_set = np.reshape(one_layer_set, shape_cur_layer)

            res.append(one_layer_set)

        return res

    def cal_TrimmedMean(self, input_weights, beta=0.1):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            one_layer_set = np.array(one_layer_set)
            print(one_layer_set)
            one_layer_results = trim_mean(one_layer_set, beta)
            print(one_layer_results)
            one_layer_results = np.reshape(one_layer_results, shape_cur_layer)

            res.append(one_layer_results)

        return res

    def cal_GeoMed(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            one_layer_set = np.array(one_layer_set).astype(float)

            one_layer_set = hdm.geomedian(one_layer_set, axis=0)

            one_layer_set = np.reshape(np.array(one_layer_set), shape_cur_layer)
            res.append(one_layer_set)

        return res

    def cal_Medoid(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            one_layer_set = np.array(one_layer_set).astype(float)

            one_layer_set = hdm.medoid(one_layer_set, axis=0)

            one_layer_set = np.reshape(np.array(one_layer_set), shape_cur_layer)
            res.append(one_layer_set)

        return res

    def cal_Krum(self, input_weights, num_machines=10, num_byz=2):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights

        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            one_layer_set = np.array(one_layer_set).astype(float)

            score = []
            num_near = num_machines - num_byz - 2
            for i, w_i in enumerate(one_layer_set):
                dist = []
                for j, w_j in enumerate(one_layer_set):
                    if i != j:
                        dist.append(np.linalg.norm(w_i - w_j) ** 2)
                dist.sort(reverse=False)
                score.append(sum(dist[0:num_near]))
            i_star = score.index(min(score))

            selected = one_layer_set[i_star]

            selected = np.reshape(np.array(selected), shape_cur_layer)
            res.append(selected)

        return res

    def cal_theroy(self, input_weights, num):
        """
        @param input_weights: 输入的各个client的权重
        @param num: 恶意攻击者个数
        """
        res = []

        for layer_idx in range(len(input_weights[0])):
            # record the shape of the current layer

            ent = []
            shape_cur_layer = input_weights[0][layer_idx].shape

            one_layer_set = [item[layer_idx].flatten() for item in input_weights]
            # print(one_layer_set)
            one_layer_set = np.array(one_layer_set).astype(float)
            for i in range(one_layer_set.shape[1]):
                dic = {}
                lis = [[] for _ in range(num)]
                min = one_layer_set[:, i].min()
                max = one_layer_set[:, i].max()
                dis = (max - min) / num
                if dis == 0:
                    break
                for j, w_j in enumerate(one_layer_set[:, i]):
                    k = (w_j - min) // dis
                    k = int(k)
                    if k == num:
                        lis[k - 1].append(j)
                    else:
                        lis[k].append(j)
                for m in range(one_layer_set.shape[0]):
                    if m not in dic.keys():
                        dic[m] = 0
                    entropy = 0
                    for n in range(len(lis)):
                        if m in lis[n]:
                            tem = lis[n][:]
                            tem.remove(m)
                            p = len(tem) / (one_layer_set.shape[0] - 1)
                            if p == 0:
                                continue
                            else:
                                entropy += -p * np.log(p)
                        else:
                            p = len(lis[n]) / (one_layer_set.shape[0] - 1)
                            if p == 0:
                                continue
                            else:
                                entropy += -p * np.log(p)
                    dic[m] += entropy
            # sorted(dic.items(), key=lambda item: item[1])
                print(dic)
                need_del = []
                mean = sum(item[1] for item in dic.items()) / len(dic)
                for item in dic.items():
                    if item[1] <= mean:
                        need_del.append(item[0])
                print(need_del)
                value = (np.delete(one_layer_set[:,i], need_del, axis=0)).mean()
                ent.append(value)
            # one_layer_set = np.delete(one_layer_set, need_del, axis=0)
            # selected = one_layer_set.mean(axis=0)
            selected = np.reshape(np.array(ent), shape_cur_layer)
            res.append(selected)

        return res
if __name__ == "__main__":
    baseline = Baseline()
    # a = np.array([ 1.0,2.0 ])
    # b = np.array( [ [ [1,2], [3,4], [4,5] ], [ [7,8], [9,10], [11,12] ], [ [13,14], [15,16], [17,18] ] ] )
    # c = np.array([ [ [ [1,2], [3,4] ] ], [ [ [5,6], [7,8] ] ] ])
    # # print(a, a.shape)
    # # print(b, b.shape)
    # # print(c, c.shape)
    # # print(b.flatten())
    # test_input = [ [a,b,c], [ a + 1, b + 1, c + 1], [ a + 2, b + 2, c + 2]]
    # print(test_input)
    # for item in baseline.cal_TrimmedMean(test_input):
    #     print(item, item.shape)
    # test_input = np.array([[1,2,3.4,4,6,2],
    #                        [1,2,3.4,4,6,2],
    #                        [1,2,3.4,4,6,2],
    #                        [1,2,3.4,4,6,2],
    #                        [12,2,3.4,4,6,2],
    #                        [4,2,3.4,4,6,2]])
    test_input = np.array([[1,3],
                           [1,4],
                           [1,3],
                           [1,16],
                           [12,3],
                           [4,4]])
    res = baseline.cal_theroy(test_input,2)
