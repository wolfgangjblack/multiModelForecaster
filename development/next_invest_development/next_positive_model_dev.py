# Databricks notebook source
NOTEBOOK_NAME = "next_positive_model_dev"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook Purpose:
# MAGIC 
# MAGIC Using pipeline and best practices, develop a model with pipeline capabilities. 
# Magic Model will use pysparks GBTClassifier to forecast the samples probablity to adopt the next_positive action

# COMMAND ----------

import pyspark.sql.functions as F
import mlflow
import mlflow.spark

from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# MAGIC %run 
# MAGIC ../../tools/model_utils

# COMMAND ----------

# MAGIC %run
# MAGIC ../../tools/model_config

# COMMAND ----------

model_name = 'next_invest_model'
model_training_config['model_name'] = model_name
model_training_config['uri'] =model_training_config['uri'] + model_name+'/'
model_training_config['run_name'] =  model_training_config['version'].split('/')[0]+'_'+model_name

# COMMAND ----------

model_training_config 

# COMMAND ----------

positive_class = spark.read.table("multimodel_forecaster.next_positive_users_positive_class")
negative_class = spark.read.table("multimodel_forecaster.next_positive_users_negative_class")
data = positive_class.union(negative_class)

# COMMAND ----------

cols_to_drop = ['uuid','next_positive_binary', 'snapshot_date', 'target']
features = [i for i in data.columns if i not in cols_to_drop]
catFeatures = features[:8]
acxiom_features = [i for i in features if '00' in i]

# COMMAND ----------

features

# COMMAND ----------

for c in acxiom_features:
  data = data.withColumn(c,
                        F.col(c).cast('float'))
data = data.withColumnRenamed("next_positive_binary", 'target')

(training_data, test_data) = data.randomSplit([0.8,0.2], seed = model_training_config['seed'])

# COMMAND ----------

with mlflow.start_run(run_name = model_training_config['run_name']) as run:
  #log parameters
  mlflow.log_params({'data': 'multimodel_forecaster.next_positive_users',
                     'positive_class_data': 'multimodel_forecaster.next_positive_users_positive_class',
                     'downsample_negative_class_data': 'multimodel_forecaster.next_positive_users_negative_class',
                     'target': 'next_positive_binary'})
  #Get Categorical features indexed
  indexers = [StringIndexer(inputCol = col, outputCol = col+"_index").fit(data) for col in catFeatures]
  
  #Assemble data into a feature vector
  vecAssembler = VectorAssembler(inputCols = [col+"_index" for col in catFeatures] + features[7:],
                               outputCol = 'features')
  
  #initalize GBTClassifier Instance
  gb = GBTClassifier(labelCol = 'target', featuresCol = 'features', seed = model_training_config['seed'], maxBins = 64)
  
  
  stages = indexers + [vecAssembler, gb]
  
  #setup Model
  pipelineModel = Pipeline(stages = stages)
  
  #Setup parameter grid for hypertuning
  paramGrid = ParamGridBuilder() \
    .addGrid(gb.maxIter, model_training_config['maxIter']) \
    .addGrid(gb.maxDepth, model_training_config['maxDepth']) \
    .addGrid(gb.stepSize, model_training_config['stepSize']) \
    .build()
  
  #Use CrossValidator to do Hypertuning
  crossval = CrossValidator(estimator=pipelineModel,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol = 'target', metricName = 'areaUnderROC'),
                          numFolds=3)
  
  #Hypertunes model
  cvModel = crossval.fit(training_data)
  
  #Select best performing model relative to BinaryClassificationEvaluator evaluating on area under ROC
  bestModel = cvModel.bestModel
  
  mlflow.spark.log_model(bestModel, 'best-model-next-positive')
  log_features_as_json(features, 'features.json')
  
  predDF = bestModel.transform(test_data)
  
  metrics = get_model_performance_dict(predDF)
  
  #log Graphs
  log_ROC_curve(predDF)
  log_f1_curve(predDF)
  log_precision_curve(predDF)
  log_recall_curve(predDF)
  
  # Log metrics
  mlflow.log_metrics(metrics)
  print(f"current model scores:\n recall: {metrics['recall']} precision:{metrics['precision']} accuracy: {metrics['accuracy']}")
  
  runID = run.info.run_id

# COMMAND ----------

print(f"the best model has a max depth of {bestModel.stages[-1].getMaxDepth()}")
print(f"the best model has a number of trees/max iter of {bestModel.stages[-1].getMaxIter()}")
print(f"the best model has a step size of {bestModel.stages[-1].getStepSize()}")


# COMMAND ----------

model_tracking_metrics = [(model_training_config['model_name'],
                         model_training_config['version'],
                         model_training_config['uri'])]

model_tracking_columns = ('model_name', 'version', 's3_uri')

model_tracking_df = spark.createDataFrame(model_tracking_metrics, schema = model_tracking_columns)
display(model_tracking_df)

# COMMAND ----------

save_model_to_s3(bestModel, model_training_config['uri'])

# COMMAND ----------

model_tracking_df.withColumn('snapshot_date',
                          F.lit(F.current_timestamp())).write.mode("append").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.training_model_records")

# COMMAND ----------

dbutils.notebook.exit('')
