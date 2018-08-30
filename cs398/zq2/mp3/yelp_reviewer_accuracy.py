from pyspark import SparkContext, SparkConf
import argparse
import json

def combine(x, y):
    return (x[0] + y[0], x[1] + y[1])

def reducer(x):
    return (x[0], round(float(x[1][0]) / float(x[1][1]), 2))

def choose(x):
    score = x.get('stars')
    business = x.get('business_id')
    user = x.get('user_id')
    if score != None and business != None and user != None:
        return (business, (score, user))
    return None

def find_user_review_accuracy(sc, reviews_filename):
    '''
    Args:
        sc: The Spark Context
        reviews_filename: Filename of the Yelp reviews JSON file to use, where each line represents a review
    Returns:
        An RDD of tuples in the following format:
            (USER_ID, AVERAGE_REVIEW_OFFSET)
            - USER_ID: The ID of the user being referenced
            - AVERAGE_REVIEW_OFFSET: The average difference between a user's review and the average restaraunt rating
    '''

    # YOUR CODE HERE
    distFile = sc.textFile(reviews_filename)
    source = distFile.map(json.loads)
    #source = sc.parallelize(source)
    source = source.map(choose)
    source = source.filter(lambda x: x != None)
    store = source.map(lambda x: (x[0], (x[1][0], 1))) #store(bxss_id, (user_id, 1))
    user = source.map(lambda x: (x[0], (x[1][1], x[1][0]))) #user(business_id, (starts, user_id))

    #store(business_id, stars_avg)
    store = store.reduceByKey(combine).map(reducer)

    #user(business_id, (user_id, stars), stars_avg)
    user = user.join(store).map(lambda x: (x[1][0][0], (float(x[1][0][1]) - x[1][1], 1)))
    result = user.reduceByKey(combine).map(reducer)
    return result.map(lambda x : (str(x[0]), x[1]))

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Yelp review data from')
    parser.add_argument('output', help='File to save RDD to')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("yelp_reviewer_accuracy")
    sc = SparkContext(conf=conf)

    results = find_user_review_accuracy(sc, args.input)
    results.saveAsTextFile(args.output)
