'''
by：wangyongkang
此处为恶意梯度更新实现
'''

import numpy as np

def same_value_attack(input, attack_value = 100):
    return np.full(input.shape, attack_value)

def sign_flipping_attack(input, attack_value = -1):
    return input * attack_value

def random_attack(input):
    # return ( (np.random.rand( *input.shape ) - 0.5) * 0.2 ) + input
    return ( np.random.normal( scale=0.3, size=input.shape ) ) + input

def backward(my_weights, weights, y = -4):
    my_weights = weights + y * (my_weights - weights)
    return my_weights


if __name__ == "__main__":
    # test_input = np.random.rand(10, 10)
    test_input = np.array([ np.random.rand(10,1), np.random.rand(1, 10), np.random.rand(2,3,4), 
                    np.random.rand(5,4,3,2,2) ])
    print("[ORIGINAL]", test_input)
    # print([  same_value_attack(item) for item in test_input ])
    test_input = np.array([ random_attack(item) for item in test_input ])
    print(test_input)
    # print("[SAME VALUE]", same_value_attack(test_input))
    # print("[SIGN FLIP]", sign_flipping_attack(test_input))
    # print("[RANDOM]", random_attack(test_input))