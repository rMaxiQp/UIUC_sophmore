from pyspark import SparkContext, SparkConf
from numpy import array, ndarray
from math import sqrt
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.clustering import KMeans, KMeansModel
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab as pl
import argparse
import json

def get_coord(line):
    if line.get("state")=="IL" :
        if line.get("city")=="Urbana" or line.get("city") == "Champaign":
            return (line.get("latitude"), line.get("longitude"))
    return None

def error(point, model):
    center = model.centers[model.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

def yelp_clustering(sc, filename):
    '''
    Args:
        sc: The Spark Context
        filename: Filename of the yelp businesses file to use, where each line represents a business
    '''

    # YOUR CODE HERE
    input_file = sc.textFile(filename).sample(False, 0.1)
    json_load = input_file.map(json.loads)

    labeled_data = json_load.map(get_coord).filter(lambda x: x != None).map(lambda x: (float(x[0]), float(x[1])))
   
    training_data, testing_data = labeled_data.randomSplit([0.7, 0.3])
    
    trail = array([x for x in range(2, 32, 2)])
    train_WSSSE = ndarray((15,))
    test_WSSSE = ndarray((15,))
    for i, j in enumerate(trail):
        model = KMeans.train(training_data, int(j), maxIterations=100)
        train_WSSSE[i] = training_data.map(lambda point: error(point, model)).reduce(lambda x,y: x +y)
        test_WSSSE[i] = testing_data.map(lambda point: error(point, model)).reduce(lambda x,y: x + y)


    plt.figure()
    plt.title("WSSSE & KMeans")
    plt.xlabel("K")
    plt.ylabel("WSSSE")
    plt.plot(trail, train_WSSSE, "b.", label = "training")
    plt.plot(trail, test_WSSSE, "g.", label = "testing")
    plt.legend()
    plt.savefig("1.png")
    plt.show()
 
    model = KMeans.train(training_data, 3, maxIterations=100, initializationMode="random") 
    
    train_preds = training_data.zip(model.predict(training_data)).map(lambda x: (x[0][0], x[0][1],x[1]))

    test_preds = testing_data.zip(model.predict(testing_data)).map(lambda x: (x[0][0], x[0][1], x[1]))
    
    plt.figure()
    pl.scatter(array(train_preds.collect())[:, 0], array(train_preds.collect())[:,1], c=array(train_preds.collect())[:,2])
    pl.savefig("train.png")    
    pl.show()

    plt.figure()
    pl.scatter(array(test_preds.collect())[:, 0], array(test_preds.collect())[:, 1], c=array(test_preds.collect())[:, 2])
    pl.show()
    pl.savefig("test.png")


if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Yelp business data from')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("yelp_clustering")
    sc = SparkContext(conf=conf)

    yelp_clustering(sc, args.input)
