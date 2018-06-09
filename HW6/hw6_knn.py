import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


class kNN(object):
    def __init__(self, k=1):
        self._train_data = None
        self._target_data = None
        self._k = k


    def fit(self, train_data, target_data):
        self._train_data = train_data
        self._target_data = target_data


    def predict(self, x):
        distances = np.array([np.linalg.norm(p - x) for p in self._train_data])
        nearest_indices = distances.argsort()[:self._k]
        nearest_labels = self._target_data[nearest_indices]
        c = Counter(nearest_labels)
        return c.most_common(1)[0][0]


def load_train_data():
    for i in range(10):
        if i==0:
            train_feature = np.loadtxt('data/digit_train{}.csv'.format(i), delimiter=',')
            train_label = np.array([i]*train_feature.shape[0])
        else:
            temp_feature = np.loadtxt('data/digit_train{}.csv'.format(i), delimiter=',')
            train_feature = np.vstack([train_feature, temp_feature])
            temp_label = np.array([i]*temp_feature.shape[0])
            train_label = np.hstack([train_label, temp_label])
            
    return train_feature, train_label


def load_test_data():
    for i in range(10):
        if i==0:
            test_feature = np.loadtxt('data/digit_test{}.csv'.format(i), delimiter=',')
            test_label = np.array([i]*test_feature.shape[0])
        else:
            temp_feature = np.loadtxt('data/digit_test{}.csv'.format(i), delimiter=',')
            test_feature = np.vstack([test_feature, temp_feature])
            temp_label = np.array([i]*temp_feature.shape[0])
            test_label = np.hstack([test_label, temp_label])
            
    return test_feature, test_label


def calc_accuracy(train_feature, train_label, test_feature, test_label, k=1):
    model = kNN(k)
    model.fit(train_feature, train_label)
    predicted_labels = []
    for feature in test_feature:
        predicted_label = model.predict(feature)
        predicted_labels.append(predicted_label)
    return accuracy_score(test_label, predicted_labels)


def load_train_data_cv(n_split=5):
    for i in range(10):
        if i==0:
            train_feature = np.loadtxt('data/digit_train{}.csv'.format(i), delimiter=',')
            train_label = np.array([i]*train_feature.shape[0])
            group_feature = np.split(train_feature, n_split)
            group_label = np.split(train_label, n_split)
        else:
            temp_feature = np.loadtxt('data/digit_train{}.csv'.format(i), delimiter=',')
            temp_group_feature = np.split(temp_feature, n_split)
            temp_label = np.array([i]*temp_feature.shape[0])
            temp_group_label = np.split(temp_label, n_split)
            
            for m in range(n_split):
                group_feature[m] = np.vstack([group_feature[m], temp_group_feature[m]])
                group_label[m] = np.hstack([group_label[m], temp_group_label[m]])
            
    return group_feature, group_label


def cross_validation(n_split=5, params=[1,2,3,4,5,10,20]):
    n_params = len(params)
    score_list = np.zeros(n_params)
    group_feature, group_label = load_train_data_cv(n_split)
    
    for j in range(n_params):
        for i in range(n_split):
            temp_group_feature = group_feature.copy()
            temp_test_feature = temp_group_feature.pop(i)
            temp_train_feature = np.vstack(temp_group_feature)
            
            temp_group_label = group_label.copy()
            temp_test_label = temp_group_label.pop(i)
            temp_train_label = np.hstack(temp_group_label)
            
            score_list[j] += calc_accuracy(temp_train_feature, temp_train_label, temp_test_feature, temp_test_label, k=params[j])/n_split

    opt_param = params[np.argmax(score_list)]
    print(score_list)
    return opt_param


def main():
    k_opt = cross_validation(n_split=5, params=[1,2,3,4,5,10,20])
    train_feature, train_label = load_train_data()
    test_feature, test_label = load_test_data()
    score = calc_accuracy(train_feature, train_label, test_feature, test_label, k=k_opt)
    print(score)


if __name__ == '__main__':
    main()