from pyspark import SparkContext, SparkConf
import argparse
import json

def greater(x, y):
    if x[1] > y[1]:
        return(x[0], x[1])
    elif x[1] == y[1]:
        if(y[0] > x[0]):
            return(y[0], y[1])
        else:
            return(x[0], x[1])
    else:
        return(y[0],y[1])

def find_engaging_reviews(sc, reviews_filename):
    '''
    Args:
        sc: The Spark Context
        reviews_filename: Filename of the Yelp reviews JSON file to use, where each line represents a review
    Returns:
        An RDD of tuples in the following format:
            (BUSINESS_ID, REVIEW_ID)
            - BUSINESS_ID: The business being referenced
            - REVIEW_ID: The ID of the review with the largest sum of "useful", "funny", and "cool" responses
                for the given business
    '''

    # YOUR CODE HERE
    distFile = sc.textFile(reviews_filename)
    source = distFile.map(json.loads) #return list of dict
    #source = sc.parallelize(source)

    source = source.map(lambda x: (x['business_id'], (x['review_id'], x['useful'] + x['funny'] + x['cool'])))
    result = source.reduceByKey(greater).reduceByKey(max).map(lambda x: (x[0], x[1][0]))
    return result

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Yelp review data from')
    parser.add_argument('output', help='File to save RDD to')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("engaging_reviews")
    sc = SparkContext(conf=conf)

    results = find_engaging_reviews(sc, args.input)
    results.saveAsTextFile(args.output)
