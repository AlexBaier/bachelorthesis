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

    def __init__(self, neighbors: int):
        self.__classifier = KNeighborsClassifier(
            algorithm='auto',
            metric='minkowski',
            n_jobs=-1,
            n_neighbors=neighbors,
            p=2,
            weights='distance',
        )  # type: KNeighborsClassifier

    def train(self, training_data: Tuple[np.array, np.array]):
        x, y = training_data
        self.__classifier.fit(x, y)

    def classify(self, unknowns: np.array)->List[str]:
        if unknowns.shape[0] == 1:
            return self.__classifier.predict(unknowns.reshape(1, -1)).tolist()
        else:
            return self.__classifier.predict(unknowns).tolist()


class KRINKNNClassifier(KNNClassifier):

    def __init__(self, neighbors: int, regularization_param: float):
        self.__x = None
        self.__y = None
        self.__nearest_neighbors = NearestNeighbors(
            metric='minkowski',
            n_neighbors=neighbors,
            n_jobs=-1,
            p=2)
        self.__neighbors = neighbors
        self.__reg = regularization_param

    def train(self, training_data: Tuple[np.array, np.array]):
        self.__x, self.__y = training_data
        self.__nearest_neighbors.fit(training_data[0])

    def classify(self, unknowns: np.array)->List[str]:
        labels = list()
        n = unknowns.shape[0]
        index_matrix = self.__nearest_neighbors.kneighbors(unknowns, return_distance=False)
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

    def __init__(self, embedding_size: int, embeddings: np.array, labels: List[str]):
        self.__embedding_size = embedding_size  # type: int
        self.__embeddings = embeddings  # type: np.array
        self.__labels = labels  # type: List[str]
        self.__nearest_neighbors = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            n_jobs=-1,
            p=2)
        self.__sgd_regressors = [SGDRegressor() for _ in range(self.__embedding_size)]

    def train(self, training_data: List[Tuple[np.array, np.array]], batch_size: int=500):
        self.__nearest_neighbors.fit(self.__embeddings)
        logging.log(level=logging.INFO, msg='fitted nearest neighbors tree on all embeddings')

        n = len(training_data)

        x, y = list(map(np.array, zip(*training_data)))
        del training_data

        batches = [batch_size for _ in range(int(n / batch_size) + 1)]
        batches[-1] = n % batch_size

        for k in range(len(batches)):
            begin = sum(batches[:k])
            end = sum(batches[:k+1])
            for target in range(self.__embedding_size):
                self.__sgd_regressors[target].partial_fit(x[begin:end], y[begin:end][:, target])
            logging.log(level=logging.INFO, msg='regression fitting progress: {}/{}'.format(k, len(batches)))

    def classify(self, unknowns: np.array)->List[str]:
        labels = list()

        y = list()
        for target in range(self.__embedding_size):
            y.append(self.__sgd_regressors[target].predict(unknowns))
        y = np.array(y).T
        # calculate similarities between projections and all embeddings
        # get indices of most similar embedding
        indexes = self.__nearest_neighbors.kneighbors(y, return_distance=False)

        for index in indexes:
            labels.append(self.__labels[index[0]])

        return labels


class PiecewiseLinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, embeddings: np.array, labels: List[str], clusters: int):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        self.__embeddings = embeddings
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__labels = labels  # type: List[str]
        self.__phi = [np.random.random((embedding_size, embedding_size)) for _ in range(clusters)]

    def train(self, training_data: List[Tuple[np.array, np.array]], batch_size: int=500,
              btol: float=1e-3, alpha: float= 1e-5, max_iter: int=50000):
        n = len(training_data)  # number of samples
        self.__kmeans.fit(self.__embeddings)

        x, y = list(map(np.array, zip(*training_data)))
        del training_data

        cluster_labels = self.__kmeans.predict(x)
        clustered_x = [list() for _ in range(self.__clusters)]
        clustered_y = [list() for _ in range(self.__clusters)]
        for i, c in enumerate(cluster_labels):
            clustered_x[c].append(x[i])
            clustered_y[c].append(y[i])
        del x
        del y
        clustered_x = list(map(np.array, clustered_x))
        clustered_y = list(map(np.array, clustered_y))

        for k in range(self.__clusters):
            old_error = 0.0
            for i in range(max_iter):
                p = np.random.permutation(clustered_x[k].shape[0])[:batch_size]
                error_vec = np.sum(clustered_x[k][p] @ self.__phi[k] - clustered_y[k][p], axis=0)
                error = 1.0 / float(batch_size) * np.sum(error_vec) ** 2
                self.__phi[k] -= alpha * np.array([error_vec for _ in range(self.__embedding_size)])
                logging.log(level=logging.INFO, msg='iter={},error={},diff={}'
                            .format(i, error, np.absolute(error - old_error)))
                if old_error < btol and error < btol:
                    logging.log(level=logging.INFO, msg='batch error lower than tolerance after 2 iterations')
                    break
                old_error = error
            logging.log(level=logging.INFO, msg='trained cluster {}/{}'.format(k, self.__clusters))

    def classify(self, unknowns: np.array)->List[str]:
        projections = np.zeros(unknowns.shape)
        labels = list()
        cluster_labels = self.__kmeans.predict(unknowns)

        # project all unknowns
        for i, c in enumerate(cluster_labels):
            projections[i] = unknowns[i] @ self.__phi[c]
        # find indices of most similar embeddings
        indexes = np.argmax(cosine_similarity(projections, self.__embeddings), axis=1)

        for index in indexes:
            labels.append(self.__labels[index])
        return labels


class TrainingFailureException(Exception):
    pass
