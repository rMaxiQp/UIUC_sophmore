from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf

import argparse

stopwords_str = 'i,me,my,myself,we,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,s,t,can,will,just,don,should,now,d,ll,m,o,re,ve,y,ain,aren,couldn,didn,doesn,hadn,hasn,haven,isn,ma,mightn,mustn,needn,shan,shouldn,wasn,weren,won,wouldn,dont,cant'
stopwords = stopwords_str.split(',')

def amazon_classification(sc, filename):
    '''
    Args:
        sc: The Spark Context
        filename: Filename of the Amazon reviews file to use, where each line represents a review
    '''

    # YOUR CODE HERE

    sqlContext = SQLContext(sc)
    dataframe = sqlContext.read.csv(filename, header='true').sample(False, 0.1)

    dataframe = dataframe.withColumn("labels", F.when(dataframe["Score"] > 2.5, 1.0).otherwise(0.0))

    training_df, test_df = dataframe.randomSplit([0.8, 0.2])

    tokenizer = Tokenizer(inputCol="Text", outputCol="words")

    filtered = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filt", stopWords = stopwords)

    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
    nb = NaiveBayes(labelCol="labels")

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])

    model = pipeline.fit(training_df)

    prediction_df = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(metricName="f1", labelCol="labels")

    result = evaluator.evaluate(prediction_df)

    labels_and_preds = prediction_df.rdd.map(lambda x: (x.labels, x.prediction))
    test = test_df.rdd.map(lambda x:(x.labels, x.prediction))
    with open('output_class_4.txt', 'w+') as f:
        f.write("Summary Stats\n")
        f.write("data split: [0.8 0.2], >2.5==>1, sample:0.1\n")
        f.write(str(MulticlassMetrics(labels_and_preds).confusionMatrix().toArray()) + '\n')
        f.write(str(MulticlassMetrics(labels_and_preds).precision())+ '\n')
        f.write(str(result) + '\n')

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Amazon review data from')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("amazon_classification")
    sc = SparkContext(conf=conf)

    amazon_classification(sc, args.input)
