# Databricks notebook source
NOTEBOOK_NAME = "model_utils"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

import matplotlib.pyplot as plt
import os 
import tempfile
import shutil
import json
from pyspark import keyword_only
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import DataFrame
from pyspark.ml import Model
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param, Params

import shap
import pandas as pd

# COMMAND ----------

def model_utils_help():
  return """
  Loads in helper functions that allow the saving of artifacts to ML flow and any functions related to modeling
  """

# COMMAND ----------

class metaDataDropper(Transformer, DefaultParamsReadable, DefaultParamsWritable):
  """
  A custom Transformer which drops all dataframe metaData - This is meant to be used AFTER an indexer but before the vector assembler
  when using a pyspark Pipeline and wanting a SHAP analysis
  """ 
  value = Param(
    Params._dummy(),
    "value",
    "value_to_ft")

  @keyword_only
  def setParams(self, outputCols=None, value=0.0):
    """
    setParams(self, outputCols=None, value=0.0)
    Sets params for this SetValueTransformer.
    """
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def _transform(self, df: DataFrame) -> DataFrame:
    df = df.rdd.toDF()
    return df

# COMMAND ----------

def save_model_to_s3(model: Model, uri: str):
  """save model to s3 - note: in our case we're saving a pipeline"""
  model \
    .write() \
    .overwrite() \
    .save(uri)

# COMMAND ----------

def log_features_as_json(featureCols: list, file_name: str):
  '''
  log_features_as_json allows the logging of feature names for model control
  '''
  tmp_path = os.path.join(tempfile.mkdtemp(), file_name)
  json.dump(featureCols, open(tmp_path, "w"))
  mlflow.log_artifact(tmp_path)
  shutil.rmtree(tmp_path, ignore_errors=True)

  return

# COMMAND ----------

# Scala version implements .roc() and .pr()
# Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
# Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

def log_ROC_curve(probability_pred_df: DataFrame) -> plt:
  """
  Generates the ROC curve using the probability as output the Logistic Regression Model. Then logs that plot as a PNG artifact to MLflow.
  """

  preds = probability_pred_df.select('target','probability') \
  .rdd.map(lambda row: (float(row['probability'][1]), float(row['target'])))
  
  points = CurveMetrics(preds).get_curve('roc')

  fig = plt.figure(figsize = (10, 7), dpi = 300)  
  x_val = [x[0] for x in points]
  y_val = [x[1] for x in points]
  plt.title('ROC')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.plot(x_val, y_val)
  
  image_path = os.path.join(tempfile.mkdtemp(), 'ROC-AUC.png')
  fig.savefig(image_path)
  mlflow.log_artifact(image_path)
  shutil.rmtree(image_path, ignore_errors=True)
  
  plt.close()
  
  return

# COMMAND ----------

def log_f1_curve(probability_pred_df: DataFrame) -> plt:
  """
    Generates the F1 curve using the probability as output the Logistic Regression Model. Then logs that plot as a PNG artifact to MLflow.
  """
  import matplotlib.pyplot as plt
  import os 
  import tempfile
  import shutil
  
  preds = probability_pred_df.select('target','probability') \
  .rdd.map(lambda row: (float(row['probability'][1]), float(row['target'])))
  
  metrics = CurveMetrics(preds)
  
  ## F1
  points = metrics.get_curve('fMeasureByThreshold')
  fig = plt.figure(figsize = (10, 7), dpi = 300)
  x_val = [x[0] for x in points]
  y_val = [x[1] for x in points]
  plt.title('F1 by Threshold')
  plt.xlabel('Threshold')
  plt.ylabel('F1')
  plt.plot(x_val, y_val)
  
  image_path = os.path.join(tempfile.mkdtemp(), 'F1_v_Threhsold.png')
  fig.savefig(image_path)
  mlflow.log_artifact(image_path)
  shutil.rmtree(image_path, ignore_errors=True)
  plt.close()
  
  return

# COMMAND ----------

def log_recall_curve(probability_pred_df: DataFrame) -> plt:
  """
   Generates the recall curve using the probability as output the Logistic Regression Model. Then logs that plot as a PNG artifact to MLflow.
  """
  import matplotlib.pyplot as plt
  import os 
  import tempfile
  import shutil
  
  preds = probability_pred_df.select('target','probability') \
  .rdd.map(lambda row: (float(row['probability'][1]), float(row['target'])))

  metrics = CurveMetrics(preds)
  points = metrics.get_curve('recallByThreshold')

  fig = plt.figure(figsize = (10, 7), dpi = 300)
  
  x_val = [x[0] for x in points]
  y_val = [x[1] for x in points]
  plt.title('Recall by Threshold')
  plt.xlabel('Threshold')
  plt.ylabel('Recall')
  plt.plot(x_val, y_val)
  
  image_path = os.path.join(tempfile.mkdtemp(), 'Recall_v_Threhsold.png')
  fig.savefig(image_path)
  mlflow.log_artifact(image_path)
  shutil.rmtree(image_path, ignore_errors=True)
  plt.close()
  return

# COMMAND ----------

def log_precision_curve(probability_pred_df: DataFrame) -> plt:
  """
  Generates the precisions curve using the probability as output the Logistic Regression Model. Then logs that plot as a PNG artifact to MLflow.
  """
  
  preds = probability_pred_df.select('target','probability') \
  .rdd.map(lambda row: (float(row['probability'][1]), float(row['target'])))
  
  metrics = CurveMetrics(preds)
  
  ##Precision
  points = metrics.get_curve('precisionByThreshold')

  fig = plt.figure(figsize = (10, 7), dpi = 300)
  x_val = [x[0] for x in points]
  y_val = [x[1] for x in points]
  plt.title('Precision by Threshold')
  plt.xlabel('Threshold')
  plt.ylabel('Precision')
  plt.plot(x_val, y_val)
  
  image_path3 = os.path.join(tempfile.mkdtemp(), 'Precision_v_Threshold.png')
  fig.savefig(image_path3)
  mlflow.log_artifact(image_path3)
  shutil.rmtree(image_path3, ignore_errors=True)
  plt.close()
  
  return

# COMMAND ----------

def prepare_training_dataset_for_shap(training_df, indexers, encoder):
  """
  using the training dataset transforms the training dataset via the indexer and encorder, then outputs the data in a pd dataframe so shap can read it
  """
  for indexer in indexers:
    training_df = indexer.fit(training_df).transform(training_df)

  training_df = encoder.fit(training_df).transform(training_df)
  
  return training_df.select(*features[len(encoder.getInputCols()):]).toPandas()

# COMMAND ----------

def log_shap_summary_plot(bestModel, training_df, indexers, encoder) -> plt:
  """
  Generates the shap summary plot. Then logs that plot as a PNG artifact to MLflow.
  """
  gbt = bestModel.stages[-1]
  
  explainer = shap.TreeExplainer(gbt)
  
  df = prepare_training_dataset_for_shap(training_df, indexers, encoder)
  
  shap_values = explainer.shap_values(df)

  
  fig = plt.figure(figsize = (10, 7), dpi = 300)
  plt.title('Shap Summary Plot')  
  shap.summary_plot(shap_values, df, feature_names = df.columns, max_display = 25)
  image_path = os.path.join(tempfile.mkdtemp(), 'Shap_Summary.png')
  fig.savefig(image_path)
  mlflow.log_artifact(image_path)
  shutil.rmtree(image_path, ignore_errors=True)
  plt.close()
  
  return

# COMMAND ----------

def get_model_performance_dict(predDF: DataFrame) -> dict:
  """gets model performance metrics and returns them as a dictionary
  Input:
    predDF: DataFrame
      - this is a dataframe which must contain 'target' and 'prediction'
  Output:
    metrics: dict
      - this is a dictionary which contains the following metrics
      TP: True Positive
      FP: False Positive
      TN: True Negative
      FN: False Negative
      recall: This is the ratio of True Positives over all Actual Positives. Can be thought of as the number of correctly identified positive class over all actual positives
        - ie. the % of 1's identified correctly out of all 1's
      precision: This is the ratio of True Positives over all predicted positives. Can be thought of as the number of correctly identified positives over all identified positives
        -ie. this is the % of 1's identified correctly out of all identified 1's
      accuracy: the number correct out of all predictions
      f1: The harmonic mean of recall and precision. Can be thought of as the balance between identifying the positive class without over predicting
  """
  TP = get_confusion_quadrant(predDF, 1, 1)

  FP = get_confusion_quadrant(predDF, 0, 1)
  
  TN = get_confusion_quadrant(predDF, 0, 0)

  FN = get_confusion_quadrant(predDF, 1, 0)

  recall = TP/(TP+FN)
  precision = TP/(TP+FP)
  acc = (TP+TN)/(TP+TN+FN+FP)
  f1 = (2*TP)/(2*TP+FP+FN)
  
  return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'recall':recall, 'precision': precision, 'accuracy':acc, 'f1':f1}

# COMMAND ----------

def get_confusion_quadrant(predDF, target_val, prediction_val):
  """This gets a quadrant of the confusion matrix for the predDF given the target and prediction value. This was made to account for the edge case where one of these is 0 
        Note: This will not work for multiclass classification as this does not take into consideration the true total.
  How to use:
  True Positive: TP = get_confusion_quadrant(predDF, 1, 1)
  False Positie: FP = get_confusion_quadrant(predDF, 0, 1)
  True Negative: TN = get_confusion_quadrant(predDF, 0,0)
  False Negative: FN = get_confusion_quadrant(predDF, 1, 0)
  """
  quadDF = predDF \
    .filter((F.col("target") == F.lit(target_val))) \
    .filter(F.col("prediction") == F.lit(prediction_val)) \
    .groupby('target', 'prediction') \
    .agg(F.count(F.lit(1)))
  
  if quadDF.count() == 0:
    return 0
  else:
    return quadDF.collect()[0][-1]

# COMMAND ----------

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    # Important: asNondeterministic requires Spark 2.3 or later
    # It can be safely removed i.e.
    # return udf(to_array_, ArrayType(DoubleType()))(col)
    # but at the cost of decreased performance
    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)

# COMMAND ----------

def get_inference_dataframe(predDF: DataFrame, model_name: str) -> DataFrame:
  """get inference dataframe from GBTClassifier Model. Inference dataframe includes the uuid, the model raw score, and the prediction
  """
  infDF = predDF \
    .withColumn("prob", to_array(col("probability"))) \
    .select(['uuid', 'prediction', "probability"] + [col("prob")[i] for i in range(2)])
  
  return infDF \
          .select("id",
                 F.col("prob[1]").alias(model_name+'_probability'),
                 F.col("prediction").alias(model_name+'_prediction'))

# COMMAND ----------

def load_model_and_evaluate(users_for_evaulation: DataFrame, model_name:str, uri:str) -> DataFrame:

  pipelineModel = PipelineModel.load(uri)
  
  predDF = pipelineModel.transform(users_for_evaulation)
  
  return get_inference_dataframe(predDF, model_name)

# COMMAND ----------

def get_model_score_deciles(users_with_model_scores: DataFrame, model_name:str) -> DataFrame:
  return users_with_model_scores \
    .withColumn(model_name+"_score_decile",
                F.ntile(10).over(Window.partitionBy().orderBy(users_with_model_scores[model_name+'_probability'])))

# COMMAND ----------

def get_binary_decisioning(users: DataFrame) -> DataFrame:
  """This function uses the columns: 'business_requirement_1' and 'business_requirement_2' to determine whether users will see these cards or not
  """
  user_level_scores = users \
    .withColumn("business_requirement_1_binary", 
               F.when(F.col("business_requirement_1_binary") == F.lit(0), F.lit(1))
                      .otherwise(F.lit(0))) \
    .withColumn("business_requirement_2_binary",
                F.when(F.col("business_requirement_2_binary") == F.lit(0), F.lit(1))
                .otherwise(F.lit(0))) \
    .select("id", "business_requirement_1_binary", 'business_requirement_2_binary')
  return user_level_scores

# COMMAND ----------

def get_transport_score(users)-> DataFrame:
  def is_immuta_enabled_cluster() -> bool:
    return bool(spark.conf.get("spark.databricks.isv.product", ''))
    
    if is_immuta_enabled_cluster():
        print('Immuta cluster detected. Setting Project Context...')
        df = spark.sql("""select immuta.set_current_project(225)""") # UPDATE
        display(df)
    else:
        print('Immuta cluster NOT detected. Skipping setting Project Context...')
        
  return users.join(spark.read.table("transport_scores") \
                    .withColumnRenamed("user_id", 'id') \
                    .withColumnRenamed('transport_adoption_prediction', 'transportprediction') \
                    .withColumnRenamed('transport_adoption_probability', 'transport_probability'), on = 'id', how = 'left')

# COMMAND ----------


