import numpy as np

class KNearestNeighbor():
    def __init__(self, k):
        self.k = k
        self.eps = 1e-8

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, num_loops=2):
        if num_loops == 2:
            distances = self.compute_distance_two_loops(X_test)
        elif num_loops ==1:
            distances = self.compute_distance_one_loop(X_test)
        else:
            distances = self.compute_distance_vectorized(X_test)
        return self.predict_labels(distances)

    def compute_distance_vectorized(self, X_test):

        # train, test
        # (train - test)^2 = train^2 - 2train*test + test^2
        X_test_squared = np.sum(X_test**2, axis=1, keepdims=True) #没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
        # print(X_test_squared.shape)
        X_train_squared = np.sum(self.X_train**2, axis=1, keepdims=True)
        # print(X_train_squared.shape)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)
        # print(two_X_test_X_train.shape)
        # print((np.sqrt(self.eps + X_test_squared - 2*two_X_test_X_train + X_train_squared.T)).shape) 90x90
        return np.sqrt(self.eps + X_test_squared - 2*two_X_test_X_train + X_train_squared.T)

    def compute_distance_one_loop(self, X_test): #updated version, now we can use one 'for loop' for calculating the distance
        num_test = X_test.shape[0]
        # print(num_test)
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            distances[i, :] = np.sqrt(self.eps + np.sum((self.X_train - X_test[i, :])**2, axis=1))

        return distances

    def compute_distance_two_loops(self, X_test):
        #Naive, inefficient way
        num_test = X_test.shape[0]
        # print(num_test)
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(self.eps + np.sum((X_test[i, :] - self.X_train[j, :])**2))
        return distances

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i,:]) #find the closest distance sorted
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            # print(k_closest_classes)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes)) #where bincount calculate the nun of each class and argmax choose the largest num_of_class.
        return y_pred

if __name__ == '__main__':

        # train = np.random.rand(10,4)
        # test = np.random.rand(1,4)
        # num_example = train.shape[0]

        # distance = np.sqrt(np.sum(test**2,axis=1,keepdims=True) + np.sum(train**2,axis=1,keepdims=True) -2*np.sum(test*train))

        # distance = np.sqrt(np.sum(test**2,axis=1,keepdims=True) + np.sum(train**2,axis=1,keepdims=True).T -2*np.dot(test, train.T))

        X = np.loadtxt('example_data/data.txt', delimiter=',')
        y = np.loadtxt('example_data/targets.txt')

        KNN = KNearestNeighbor(k=3)
        KNN.train(X, y) # we are now applying X_test and X_train both with X
        # we are doing the prediction on the exact same data

        # KNN.train(train, np.zeros((num_example)))
        y_pred = KNN.predict(X, num_loops=0)
        # y_pred = KNN.predict(test, num_loops=1)

        # corr_distance = KNN.compute_distance_vectorized(test)
        print(f"Accuracy: {sum(y_pred == y) / y.shape[0]}")
        # print(f'The difference is : {np.sum(np.sum((corr_distance - distance)**2))}')