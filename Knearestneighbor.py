import numpy as np

class KNearestNeighbor():
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        distances = self.compute_distance(X_test)
        return self.predict_labels(distances)

    def compute_distance(self, X_test):
        #Naive, inefficient way
        num_test = X_test.shape[0]
        # print(num_test)
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :])**2))
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
        X = np.loadtxt('example_data/data.txt', delimiter=',')
        y = np.loadtxt('example_data/targets.txt')

        KNN = KNearestNeighbor(k=3)
        KNN.train(X, y) # we are now applying X_test and X_train both with X
        # we are doing the prediction on the exact same data
        y_pred = KNN.predict(X)

        print(f"Accuracy: {sum(y_pred == y) / y.shape[0]}")