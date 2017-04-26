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


class PiecewiseLinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, clusters: int, sgd_iter: int=5,
                 n_jobs: int=-1):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        # fit predict shape: (_, embedding_size)
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__labels = None  # type: List[str]
        # nearest neighbors shape: (_, embedding_size)
        self.__nearest_neighbors_superclasses = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )
        self.__sgd_iter = sgd_iter
        # fit predict shape (_, embedding_size+1)
        self.__sgd_regressors = [[SGDRegressor(n_iter=self.__sgd_iter) for _ in range(self.__embedding_size)]
                                 for _ in range(self.__clusters)]

    def train(self, training_data: Tuple[List[Tuple[np.array, np.array]], List[str]]):
        """
        Train the piecewise linear projection on subclass-superclass embeddings pairs.
        :param training_data: (training_samples, superclass_labels). training_samples is a list of subclass-superclass
            embeddings. superclass_labels are the corresponding labels (IDs) of the superclasses.
        :return: PiecewiseLinearProjectionClassifier is trained as side-effect, nothing is returned.
        """
        training_samples, self.__labels = training_data

        n = len(training_samples)
        print(n)

        x, y = list(map(np.array, zip(*training_samples)))
        del training_samples

        mod_x = np.append(x, np.ones((n, 1)), axis=1)

        # fit nearest neighbors on word embeddings
        self.__nearest_neighbors_superclasses.fit(y)
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
        _, indexes = self.__nearest_neighbors_superclasses.kneighbors(projections, return_distance=True)
        # retrieve the label of each superclass
        labels = list()
        for i in range(indexes.shape[0]):
            labels.append(self.__labels[indexes[i][0]])

        return labels
