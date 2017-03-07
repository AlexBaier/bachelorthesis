import abc
from typing import List, Tuple

from gensim.models import Word2Vec
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import DistanceMetric, KNeighborsClassifier


class Classifier(object, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data):
        pass

    @abc.abstractclassmethod
    def classify(self, unknown: np.array)->str:
        pass


class DistanceNearestNeighbors(Classifier):

    def __init__(self, neighbors=5):
        self.__classifier = KNeighborsClassifier(
            n_neighbors=neighbors,
            weights='distance',
            p=DistanceMetric.get_metric(cls='pyfunc', args=cosine_distances)
        )  # type: KNeighborsClassifier

    def train(self, training_data: Tuple(np.array, np.array)):
        x, y = training_data
        self.__classifier.fit(x, y)

    def classify(self, unknown: np.array)->str:
        return self.__classifier.predict(unknown)


class KRINearestNeighbors(Classifier):
    pass


class LinearProjection(Classifier):

    def __init__(self, embedding_size: int, model: Word2Vec):
        self.__embedding_size = embedding_size  # type: int
        self.__model = model  # type: Word2Vec
        self.__phi = np.random.rand(embedding_size, embedding_size)  # type: np.array

    def train(self, training_data: List[Tuple[np.array, np.array]]):
        n = len(training_data)  # number of samples

        def f(phi):
            error = 0.0
            for x, y in training_data:
                error += np.square(np.linalg.norm(phi*x - y))
            return 1.0/n * error

        result = minimize(fun=f, x0=self.__phi)

        if not result.success:
            raise TrainingFailureException(result.message)

        self.__phi = result.x

    def classify(self, unknown: np.array)->str:
        return self.__model.similar_by_vector(self.__phi * unknown, topn=1)[0][0]


class PiecewiseLinearProjection(Classifier):

    def __init__(self, embedding_size: int, model: Word2Vec, clusters: int):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__model = model  # type: Word2Vec
        self.__phi = [np.random.rand(embedding_size, embedding_size) for _ in range(clusters)]

    def train(self, training_data: List[Tuple[np.array, np.array]], refit_clusters: bool=False):
        n = len(training_data)  # number of samples
        if not refit_clusters:
            self.__kmeans.fit(np.array([v for t in training_data for v in t]))
        for k in range(self.__clusters):

            def f(phi):
                error = 0.0
                for x, y in training_data:
                    if self.__kmeans.predict(x) == k:
                        error += np.square(np.linalg.norm(phi*x - y))
                return 1.0/n * error

            result = minimize(fun=f, x0=self.__phi[k])

            if not result.success:
                raise TrainingFailureException(result.message)

            self.__phi[k] = result.x

    def classify(self, unknown: np.array)->str:
        y = self.__phi[self.__kmeans.predict(unknown)[0]]*unknown
        return self.__model.similar_by_vector(y)[0][0]


class TrainingFailureException(Exception):
    pass
