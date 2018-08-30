from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import argparse

def setup_table(sc, sqlContext, users_filename, businesses_filename, reviews_filename):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
        users_filename: Filename of the Yelp users file to use, where each line represents a user
        businesses_filename: Filename of the Yelp checkins file to use, where each line represents a business
        reviews_filename: Filename of the Yelp reviews file to use, where each line represents a review
    Parse the users/checkins/businesses files and register them as tables in Spark SQL in this function
    '''
    df = sqlContext.read.json(reviews_filename)
    dt = sqlContext.read.json(businesses_filename)
    db = sqlContext.read.json(users_filename)
    sqlContext.registerDataFrameAsTable(df, 'review')
    sqlContext.registerDataFrameAsTable(dt, 'business')
    sqlContext.registerDataFrameAsTable(db, 'users')

def query_1(sc, sqlContext):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
    Returns:
        An int: the maximum number of "funny" ratings left on a review created by someone who started "yelping" in 2012
    '''
    return sqlContext.sql("SELECT MAX(review.funny) "
                          "FROM review "
                          "INNER JOIN users "
                          "ON review.user_id = users.user_id "
                          "WHERE users.yelping_since >= '2012' AND users.yelping_since < '2013'").collect()[0][0]

def query_2(sc, sqlContext):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
    Returns:
        A list of strings: the user ids of anyone who has left a 1-star review, has created more than 250 reviews,
            and has left a review at a business in Champaign, IL
    '''
    list_l = sqlContext.sql("SELECT DISTINCT users.user_id "
                            "FROM users "
                            "INNER JOIN review "
                            "ON review.user_id = users.user_id "
                            "INNER JOIN business "
                            "ON business.business_id = review.business_id "
                            "WHERE business.city='Champaign'AND business.state='IL' AND users.review_count > 250 AND review.stars = 1  ").collect()
    retval = []
    for i in list_l:
        retval.append(i[0])
    return retval

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('users', help='File to load Yelp user data from')
    parser.add_argument('businesses', help='File to load Yelp business data from')
    parser.add_argument('reviews', help='File to load Yelp review data from')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("jaunting_with_joins")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    setup_table(sc, sqlContext, args.users, args.businesses, args.reviews)

    result_1 = query_1(sc, sqlContext)
    result_2 = query_2(sc, sqlContext)

    print("-" * 15 + " OUTPUT " + "-" * 15)
    print("Query 1: {}".format(result_1))
    print("Query 2: {}".format(result_2))
    print("-" * 30)
