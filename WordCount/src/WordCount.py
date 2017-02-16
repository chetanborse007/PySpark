#!/usr/bin/python
'''
@File:           WordCount.py
@Description:    PySpark application for finding word frequency.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created_on:     02/14/2017
@Usage:          spark-submit --master yarn --deploy-mode client 
                    src/WordCount.py
                    -i "input/WordCount"
                    -o "output/Spark/WordCount"
@python_version: 2.7
===============================================================================
'''


import argparse
import warnings

from pyspark import SparkContext, SparkConf

warnings.filterwarnings("ignore")


def Tokenizer(input):
    if not input:
        return
    
    tokens = input.split()
    tokens = map(lambda x: x.strip(), tokens)
    tokens = filter(None, tokens)
    
    if not tokens:
        return
    
    for token in tokens:
        yield (token, 1)


def WordCount(**args):
    """
    Entry point for PySpark application for finding word frequency.
    """
    # Read arguments
    input  = args['input']
    output = args['output']

    # Create SparkContext object
    conf = SparkConf()
    conf.setAppName("WordCount")
    conf.set('spark.driver.memory', '4g')
#     conf.set('spark.executor.cores', '4')
#     conf.set('spark.executor.memory', '10g')
    conf.set('spark.kryoserializer.buffer.max', '2000')
#     conf.set('spark.shuffle.memoryFraction', '0.5')
    conf.set('spark.yarn.executor.memoryOverhead', '4096')
    
    sc = SparkContext(conf=conf)

    # Read input from HDFS and load it into memory
    # PySpark mainly deals with RDDs, which is 'input' here
    input = sc.textFile(input)

    # Tokenize input
    input = input.flatMap(Tokenizer) \
                 .filter(lambda x: x != None)
    input = input.partitionBy(10).cache()

    # Find word frequency
    input = input.reduceByKey(lambda x, y: x + y)

    # Sort words by frequency
    input = input.sortBy(lambda x: x[1], \
                         ascending=False, \
                         numPartitions=1)
    
    # Save output
    input = input.map(lambda x: x[0] + '\t' + x[1])
    input.saveAsTextFile(output)

    # Shut down SparkContext
    sc.stop()


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Word Count Application',
                                     prog='spark-submit --master yarn --deploy-mode client \
                                           src/WordCount.py \
                                           -i <input> \
                                           -o <output>')

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input for Word Count Application.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output of Word Count Application.")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run Word Count Application
    WordCount(**args)

