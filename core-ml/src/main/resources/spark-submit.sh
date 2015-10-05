#!/bin/bash

/home/hduser/spark/bin/./spark-submit --class io.summalabs.confetti.ml.HIPIImageLoader --master local[2] /home/hduser/workspace/core-ml-1.0-SNAPSHOT-jar-with-dependencies.jar hdfs://localhost:9000/user/hduser/hib/1
