import abc
import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
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
        self.__phi = np.random.random((embedding_size, embedding_size))

    def train(self, training_data: List[Tuple[np.array, np.array]],
              parts: int=8, tol: float=1e-3, eps: float= 1e-5, max_iter: int=250000):
        n = len(training_data)  # number of samples
        if parts <= n:
            part_distribution = [int(n / parts) for _ in range(parts)]
            part_distribution[parts-1] = int(n / parts + n % parts)
        else:
            part_distribution = [int(n)]

        x, y = list(map(np.array, zip(*training_data)))
        del training_data

        def f(phi):
            # minimize flattens input matrix, return it to matrix shape
            phi = np.reshape(phi, (self.__embedding_size, self.__embedding_size))
            error = 0.0
            for c in range(parts):
                begin = sum(part_distribution[:c])
                end = sum(part_distribution[:c+1])
                error += np.square(np.linalg.norm(np.sum(x[begin:end] @ phi - y[begin:end], axis=0)))
            return 1.0/float(n) * error

        result = minimize(fun=f, x0=self.__phi, method='BFGS',
                          options={'disp': False, 'gtol': tol, 'eps': eps, 'maxiter': max_iter, 'return_all': False,
                                   'norm': np.inf})

        if not result.success:
            raise TrainingFailureException(result.message)

        # minimize flattens input matrix, return it to matrix shape
        self.__phi = np.reshape(result.x, (self.__embedding_size, self.__embedding_size))

    def classify(self, unknowns: np.array)->List[str]:
        labels = list()
        # project all unknowns
        y = unknowns @ self.__phi
        # calculate similarities between projections and all embeddings
        # get indices of most similar embedding
        indexes = np.argmax(cosine_similarity(y, self.__embeddings), axis=1)

        for index in indexes:
            labels.append(self.__labels[index])
        return labels


class PiecewiseLinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, embeddings: np.array, labels: List[str], clusters: int):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        self.__embeddings = embeddings
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__labels = labels  # type: List[str]
        self.__phi = [np.random.random((embedding_size, embedding_size)) for _ in range(clusters)]

    def train(self, training_data: List[Tuple[np.array, np.array]],
              tol: float=1e-3, eps: float= 1e-5, max_iter: int=250000):
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

            def f(phi):
                # minimize flattens input matrix, return it to matrix shape
                phi = np.reshape(phi, (self.__embedding_size, self.__embedding_size))
                error = np.square(np.linalg.norm(clustered_x[k] @ phi - clustered_y[k]))
                return 1.0/n * error

            result = minimize(fun=f, x0=self.__phi[k], method='BFGS',
                              options={'disp': False, 'gtol': tol, 'eps': eps, 'maxiter': max_iter, 'return_all': False,
                                       'norm': np.inf})

            if not result.success:
                raise TrainingFailureException(result.message)

            self.__phi[k] = np.reshape(result.x, (self.__embedding_size, self.__embedding_size))

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
