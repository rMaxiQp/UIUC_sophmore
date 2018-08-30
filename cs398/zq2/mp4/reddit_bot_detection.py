from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import argparse
import sys
import json
import difflib

def group(x):
    try:
        k = json.loads(x)
        return ((k['author'], 1), (k['text'], 1))
    except:
        return None

def freq(x, y):
    seq = difflib.SequenceMatcher(None, x[1][0], y[1][0])
    if(seq.ratio() > 0.5):
        x[1][1] += 1
    return (x[0], (x[1][0], x[1][1]))


def detect_reddit_bots(sc, input_dstream):
    '''
    Args:
        sc: the SparkContext
        input_dstream: The discretized stream (DStream) of input
    Returns:
        A DStream containing the list of all detected bot usernames
    '''

    # YOUR CODE HERE
    WINDOW_LENGTH = 20
    INTERVAL = 10

    input_list = input_dstream.map(group).filter(lambda x: x != None)
    transform = input_list.reduceByKeyAndWindow(lambda x, y: ((x[0], x[0][1] + y[0][1]), x[1]), lambda x, y: ((x[0], x[0][1] - y[0][1]), x[1]), WINDOW_LENGTH, INTERVAL)
    potential = transform.filter(lambda x: x[0][1] > WINDOW_LENGTH).reduceByKey(freq).map(lambda x: (x[0], x[1][1]))
    result = potential.filter(lambda x: x[1] > WINDOW_LENGTH)
    return result

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stream', help='Stream URL to pull data from')
    parser.add_argument('--input_stream_port', help='Stream port to pull data from')

    parser.add_argument('output', help='Directory to save DStream results to')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("reddit_bot_detection")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)
    ssc.checkpoint('streaming_checkpoints')

    if args.input_stream and args.input_stream_port:
        dstream = ssc.socketTextStream(args.input_stream,
                                       int(args.input_stream_port))
        results = detect_reddit_bots(sc, dstream)
    else:
        print("Need input_stream and input_stream_port")
        sys.exit(1)

    results.saveAsTextFiles(args.output)
    results.pprint()

    ssc.start()
    ssc.awaitTermination()
