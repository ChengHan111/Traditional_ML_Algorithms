import numpy as np

class NaiveBayes():
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape #90 for num_examples, 2 for num_features
        self.num_classes = len(np.unique(y)) # 1,2,3,4 find the unique number, in this section we have 3
        self.eps = 1e-6

    def fit(self, X):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c] #pick for specific class
            # get mean, variance and prior probability for given class.

            self.classes_mean[str(c)] = np.mean(X_c, axis=0) # zong xiang ji suan
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0]/self.num_examples #X.shape[0]

    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes)) # 90 x 3

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            # print(self.classes_variance[str(c)])
            probs_c = self.density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior) #since we have done the log calculation, the original times now becomes plus

        return np.argmax(probs, axis=1) # predict the highest # heng xiang ji suan

    def density_function(self, x, mean, sigma):
        # calculate probability from gaussian density function
        # print('???',sigma.shape)
        const = -self.num_features/2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps)) #since sigma is a 1 x 2 matrix, we can simply use sum over here.
        probs = 0.5 * np.sum(np.power(x - mean, 2)/(sigma + self.eps), 1)
        return const - probs

if __name__ == '__main__':
    X = np.loadtxt('example_data/data.txt', delimiter=',')
    y = np.loadtxt('example_data/targets.txt') - 1

    print(X.shape)
    print(y.shape)

    NB = NaiveBayes(X, y)
    NB.fit(X)
    y_pred = NB.predict(X)

    print(f'Accuracy: {sum(y_pred == y) / X.shape[0]}')
