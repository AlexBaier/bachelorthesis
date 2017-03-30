import abc
import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.metrics.pairwise import cosine_similarity
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


class KRIKNNClassifier(KNNClassifier):

    def __init__(self, neighbors: int, regularization_param: float, n_jobs: int=-1):
        self.__x = None
        self.__y = None
        self.__nearest_neighbors = NearestNeighbors(
            metric='minkowski',
            n_neighbors=neighbors,
            p=2,
            n_jobs=n_jobs
        )
        self.__neighbors = neighbors
        self.__reg = regularization_param

    def train(self, training_data: Tuple[np.array, np.array]):
        self.__x, self.__y = training_data
        self.__nearest_neighbors.fit(training_data[0])

    def classify(self, unknowns: np.array)->List[str]:
        labels = list()
        n = unknowns.shape[0]
        _, index_matrix = self.__nearest_neighbors.kneighbors(unknowns, return_distance=True)
        logging.log(level=logging.INFO, msg='KRI-kNN: found {}-nearest neighbors for {} unknowns'
                    .format(self.__neighbors, n))
        for i in range(n):
            indexes = index_matrix[i]
            neighbors = np.array([self.__x[idx] for idx in indexes])
            S = cosine_similarity(neighbors, neighbors)
            s = cosine_similarity(unknowns[i].reshape(1, -1), neighbors)[0]

            def f(w):
                return 0.5 * (w.T @ S @ w) - (s.T @ w) + (0.5 * self.__reg) * (w.T @ w)
            constraints = ({'type': 'ineq', 'fun': lambda w: 1 if (1 - np.sum(w)) > 0 and np.alltrue(w > 0) else -1})
            w0 = np.random.random(self.__neighbors)
            w0 /= np.linalg.norm(w0)
            result = minimize(f, x0=w0, method='COBYLA', constraints=constraints)

            if not result.success:
                logging.log(level=logging.INFO,
                            msg='KRI-kNN: weights did not converge for unknowns[{}], use similarity-based weights'
                            .format(i))
                weights = s/np.linalg.norm(s)
            else:
                weights = result.x
            votes = dict()
            for neighbor, idx in enumerate(indexes):
                if not votes.get(self.__y[idx], None):
                    votes[self.__y[idx]] = 0.0
                votes[self.__y[idx]] += weights[neighbor]
            labels.append(max(votes.items(), key=lambda t: t[1])[0])
            logging.log(level=logging.INFO, msg='KRI-kNN: classification progress: {}%'.format(100.0*float(i)/n))
        return labels


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

    def train(self, training_data: List[Tuple[np.array, np.array]], batch_size: int=500):
        self.__nearest_neighbors.fit(self.__embeddings)

        n = len(training_data)

        x, y = list(map(np.array, zip(*training_data)))
        del training_data

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
        labels = list()

        y = list()
        for target in range(self.__embedding_size):
            y.append(self.__sgd_regressors[target].predict(unknowns))
        y = np.array(y).T
        # calculate similarities between projections and all embeddings
        # get indices of most similar embedding
        _, indexes = self.__nearest_neighbors.kneighbors(y, return_distance=True)

        for index in indexes:
            labels.append(self.__labels[index[0]])

        return labels


class PiecewiseLinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, embeddings: np.array, labels: List[str], clusters: int, sgd_iter: int=5,
                 n_jobs: int=-1):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        self.__embeddings = embeddings
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__labels = labels  # type: List[str]
        self.__nearest_neighbors_embeddings = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )
        self.__sgd_iter = sgd_iter
        self.__sgd_regressors = [[SGDRegressor() for _ in range(self.__embedding_size)] for _ in range(self.__clusters)]

    def train(self, training_data: List[Tuple[np.array, np.array]], batch_size: int=500):
        x, y = list(map(np.array, zip(*training_data)))
        del training_data

        # compute clusters based on x (inputs)
        cluster_labels = self.__kmeans.fit_predict(x)
        logging.log(level=logging.INFO, msg='finished fit and predict of clusters, clusters={}'.format(self.__clusters))
        clustered_x = [list() for _ in range(self.__clusters)]
        clustered_y = [list() for _ in range(self.__clusters)]
        for i, c in enumerate(cluster_labels):
            clustered_x[c].append(x[i])
            clustered_y[c].append(y[i])

        clustered_x = list(map(np.array, clustered_x))
        clustered_y = list(map(np.array, clustered_y))
        logging.log(level=logging.INFO, msg='sorted training samples into clusters')

        # use sgd regression on <self.__embedding_size> targets to learn <self.__clusters> projection matrices
        for c in range(self.__clusters):
            n = len(clustered_x[c])

            batches = [batch_size for _ in range(int(n / batch_size) + 1)]
            batches[-1] = n % batch_size

            for i in range(self.__sgd_iter):
                p = np.random.permutation(n)
                for k in range(len(batches)):
                    begin = sum(batches[:k])
                    end = sum(batches[:k + 1])
                    for target in range(self.__embedding_size):
                        self.__sgd_regressors[c][target].partial_fit(
                            clustered_x[c][p[begin:end]],
                            clustered_y[c][p[begin:end]][:, target])
                logging.log(level=logging.INFO, msg='sgd regression: cluster={}/{}, iter={}/{}'
                            .format(c+1, self.__clusters, i+1, self.__sgd_iter))
        logging.log(level=logging.INFO, msg='finished training projections')

        # fit nearest neighbors on word embeddings
        self.__nearest_neighbors_embeddings.fit(self.__embeddings)
        logging.log(level=logging.INFO, msg='finished fitting all-embeddings nearest neighbor')

    def classify(self, unknowns: np.array)->List[str]:
        labels = list()

        # only one sample, reshape to prevent warnings
        if unknowns.shape[0] == 1:
            unknowns = unknowns.reshape(1, -1)

        # assign clusters to each unknown
        cluster_labels = self.__kmeans.predict(unknowns)
        logging.log(level=logging.INFO, msg='clustered unknowns, clusters={}'.format(self.__clusters))

        # compute projection for each unknown
        projections = np.zeros(unknowns.shape)
        for idx, cluster in enumerate(cluster_labels):
            for target in range(self.__embedding_size):
                projections[idx][target] = self.__sgd_regressors[cluster][target].predict(unknowns[idx].reshape(1, -1))
        logging.log(level=logging.INFO, msg='computed projections for all unknowns')

        del cluster_labels

        # find nearest neighbor to each projection
        _, indexes = self.__nearest_neighbors_embeddings.kneighbors(projections, return_distance=True)
        logging.log(level=logging.INFO, msg='found corresponding superclass embeddings')

        del projections

        # find corresponding label of each superclass
        for index in indexes:
            label = self.__labels[index[0]]
            labels.append(label)

        return labels
