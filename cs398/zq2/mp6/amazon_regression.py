from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import argparse
import csv


def pasePoint(line):
    vals = [float(x) for x in line.replace(",", " ").split(" ")]
    return LabeledPoint(float(values[0]), values[1:])

def get_ratio(x):
    f = list(csv.reader([x]))[0]
    try:
        return (float(f[4])/float(f[5]), f[9])
    except:
        return None

    '''
    def printer(x):
        print(x)

    def format_prediction(x):
        return "actual: {0}, predicted: {1}".format(x[0], float(x[1]))
    '''

def produce_tfidf(x):
    tf = HashingTF().transform(x)
    idf = IDF(minDocFreq=20).fit(tf)
    tfidf = idf.transform(tf)
    return tfidf

def amazon_regression(sc, filename):
    '''
    Args:
        sc: The Spark Context
        filename: Filename of the Amazon reviews file to use, where each line represents a review
    '''

    # YOUR CODE HERE

    # Load in reviews
    reviews = sc.textFile(filename).sample(False, 0.1)

    # Tokenize and weed out bad data
    labeled_data = (reviews.map(get_ratio)
                            .filter(lambda x: x != None)
                            .mapValues(lambda x: x.split()))

    labels = labeled_data.keys()

    tfidf = produce_tfidf(labeled_data.map(lambda x: x[1]))
    labeled_points = (labels.zip(tfidf)
                     .map(lambda x: LabeledPoint(x[0], x[1])))

    # Do a random split so we can test our model on non-trained data
    training, test = labeled_points.randomSplit([0.7, 0.3])

    # Train our model
    model = LinearRegressionWithSGD.train(training, iterations=1000, step = 0.000001)

    # Use our model to predict
    train_preds = training.map(lambda p: (p.label, model.predict(p.features)))
    train_MSE = train_preds.map(lambda vp : (vp[0] - vp[1]) ** 2).reduce(lambda x, y: x + y) / train_preds.count()

    test_preds = test.map(lambda x: (x.label, model.predict(x.features)))
    test_MSE = test_preds.map(lambda vp : (vp[0] - vp[1]) ** 2).reduce(lambda x, y: x + y) / test_preds.count()

    # Ask PySpark for some metrics on how our model predictions performed
    trained_metrics = RegressionMetrics(train_preds.mapValues(float))
    test_metrics = RegressionMetrics(test_preds.mapValues(float))

    print(trained_metrics.explainedVariance)
    print(trained_metrics.rootMeanSquaredError)
    print(str(tain_MSE))
    
    print(test_metrics.explainedVariance)
    print(test_metrics.rootMeanSquaredError)
    print(str(test_MSE))
    
if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Amazon review data from')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("amazon_regression")
    sc = SparkContext(conf=conf)

    amazon_regression(sc, args.input)
