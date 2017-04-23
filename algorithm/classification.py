import abc
import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class Classifier(object, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data):
        pass

    @abc.abstractclassmethod
    def classify(self, unknowns: np.array)->List[str]:
        pass


class KNNClassifier(Classifier, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data: Tuple[np.array, np.array]):
        pass


class ProjectionClassifier(Classifier, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data: List[Tuple[np.array, np.array]]):
        pass


class DistanceKNNClassifier(KNNClassifier):

    def __init__(self, neighbors: int, n_jobs: int=-1):
        self.__classifier = KNeighborsClassifier(
            algorithm='auto',
            metric='minkowski',
            n_neighbors=neighbors,
            p=2,
            weights='distance',
            n_jobs=n_jobs
        )  # type: KNeighborsClassifier

    def train(self, training_data: Tuple[np.array, np.array]):
        x, y = training_data
        self.__classifier.fit(x, y)

    def classify(self, unknowns: np.array)->List[str]:
        if unknowns.shape[0] == 1:
            return self.__classifier.predict(unknowns.reshape(1, -1)).tolist()
        else:
            return self.__classifier.predict(unknowns).tolist()


class LinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, embeddings: np.array, labels: List[str], sgd_iter: int=5, n_jobs: int=-1):
        self.__embedding_size = embedding_size  # type: int
        self.__embeddings = embeddings  # type: np.array
        self.__labels = labels  # type: List[str]
        self.__nearest_neighbors = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )
        self.__sgd_iter = sgd_iter
        self.__sgd_regressors = [SGDRegressor() for _ in range(self.__embedding_size)]

    def train(self, training_data: List[Tuple[np.array, np.array]], batch_size: int=1000):
        self.__nearest_neighbors.fit(self.__embeddings)

        n = len(training_data)

        x, y = list(map(np.array, zip(*training_data)))
        del training_data

        x = np.append(x, np.ones((n, 1)), axis=1)
        y = np.append(y, np.ones((n, 1)), axis=1)

        batches = [batch_size for _ in range(int(n / batch_size) + 1)]
        batches[-1] = n % batch_size

        for i in range(self.__sgd_iter):
            p = np.random.permutation(n)
            for k in range(len(batches)):
                begin = sum(batches[:k])
                end = sum(batches[:k+1])
                for target in range(self.__embedding_size):
                    self.__sgd_regressors[target].partial_fit(x[p[begin:end]], y[p[begin:end]][:, target])
                logging.log(level=logging.INFO,
                            msg='iter={}/{},batch progress: {}/{}'.format(i+1, self.__sgd_iter, k+1, len(batches)))

    def classify(self, unknowns: np.array)->List[str]:
        unknowns = np.append(unknowns, np.ones((unknowns.shape[0], 1)), axis=1)

        y = list()
        for target in range(self.__embedding_size):
            y.append(self.__sgd_regressors[target].predict(unknowns))
        y = np.array(y).T
        # calculate similarities between projections and all embeddings
        # get indices of most similar embedding
        _, indexes = self.__nearest_neighbors.kneighbors(y, return_distance=True)

        labels = list()
        for index in indexes:
            labels.append(self.__labels[index[0]])

        return labels


class PiecewiseLinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, embeddings: np.array, labels: List[str], clusters: int, sgd_iter: int=5,
                 n_jobs: int=-1):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        self.__embeddings = embeddings
        # fit predict shape: (_, embedding_size)
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__labels = labels  # type: List[str]
        # nearest neighbors shape: (_, embedding_size)
        self.__nearest_neighbors_embeddings = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )
        self.__sgd_iter = sgd_iter
        # fit predict shape (_, embedding_size+1)
        self.__sgd_regressors = [[SGDRegressor(n_iter=self.__sgd_iter) for _ in range(self.__embedding_size)]
                                 for _ in range(self.__clusters)]

    def train(self, training_data: List[Tuple[np.array, np.array]], batch_size: int=500):
        n = len(training_data)
        print(n)

        x, y = list(map(np.array, zip(*training_data)))
        del training_data

        mod_x = np.append(x, np.ones((n, 1)), axis=1)

        # fit nearest neighbors on word embeddings
        self.__nearest_neighbors_embeddings.fit(self.__embeddings)
        logging.log(level=logging.INFO, msg='finished fitting all-embeddings nearest neighbor')

        # compute clusters based on x (inputs)
        cluster_labels = self.__kmeans.fit_predict(x)
        logging.log(level=logging.INFO, msg='finished fit and predict of clusters, clusters={}'.format(self.__clusters))

        # fit nearest neighbor on cluster centers
        clustered_x = [list() for _ in range(self.__clusters)]
        clustered_y = [list() for _ in range(self.__clusters)]
        for i, c in enumerate(cluster_labels):
            clustered_x[c].append(mod_x[i])
            clustered_y[c].append(y[i])

        clustered_x = list(map(np.array, clustered_x))
        clustered_y = list(map(np.array, clustered_y))
        logging.log(level=logging.INFO, msg='sorted training samples into clusters')

        for c in range(self.__clusters):
            for target in range(self.__embedding_size):
                self.__sgd_regressors[c][target].fit(clustered_x[c], clustered_y[c][:, target])
            logging.log(level=logging.INFO, msg='sgd regression: cluster={}/{}'.format(c+1, self.__clusters))
        logging.log(level=logging.INFO, msg='finished training projections')

    def classify(self, unknowns: np.array)->List[str]:
        # add 1 row to unknowns matrix for translation
        mod_unknowns = np.append(unknowns, np.ones((unknowns.shape[0], 1)), axis=1)
        # find most appropiate cluster for all unknowns
        clusters = self.__kmeans.predict(unknowns)
        # compute projections for all unknowns
        projections = np.zeros((unknowns.shape[0], self.__embedding_size))
        for idx, cluster in enumerate(clusters):
            for target in range(self.__embedding_size):
                projections[idx][target] = self.__sgd_regressors[cluster][target]\
                    .predict(mod_unknowns[idx].reshape(1, -1))
        # find closest class to each projection => superclass
        _, indexes = self.__nearest_neighbors_embeddings.kneighbors(projections, return_distance=True)
        # retrieve the label of each superclass
        labels = list()
        for i in range(indexes.shape[0]):
            labels.append(self.__labels[indexes[i][0]])

        return labels
