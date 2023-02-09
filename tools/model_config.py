# Databricks notebook source
NOTEBOOK_NAME = "model_config"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

def model_config_help():
  return """This config notebook is meant to control both the training and the deploying for models
  To update the model version in training, change the training_version parameter
  One can also update the model_training_config to change the hypertuning.
  
  To change what models are called in inference, see the model_run_config.
  
  Currently, this config is set to always call the most recent model of each type to run
  """

# COMMAND ----------

model_tracking_df = spark.read.table("training_model_records")

# COMMAND ----------

training_version = 'V1p0/'
training_directory = 'models/'

# COMMAND ----------

model_training_config = {'maxDepth': [2, 5, 10],
                        'maxIter': [5, 20, 100],
                        'stepSize':[0.01, 0.1],
                         'seed': 1234,
                         'version': training_version,
                        'uri': 's3://mydir/'+training_directory+training_version}

# COMMAND ----------

model_run_config = {
  'Recurring_Action':{'target': 'recurring_action_binary', 
                 'feature_key': 'recurring_action',
                'uri': str((model_tracking_df \
       .withColumn('row', 
                  F.row_number().over(Window.partitionBy('model_name').orderBy(F.col("snapshot_date").desc()))) \
       .filter((F.col("model_name") == F.lit('recurringAction_model')) &
        (F.col("row") == F.lit(1))) \
       .select("s3_uri").collect()[0][-1]))},
  
  'Small_Action':{'target': 'small_action_binary', 
                 'feature_key': 'small_action',
                'uri':  str((model_tracking_df \
       .withColumn('row', 
                  F.row_number().over(Window.partitionBy('model_name').orderBy(F.col("snapshot_date").desc()))) \
       .filter((F.col("model_name") == F.lit('smallAction_model')) &
        (F.col("row") == F.lit(1))) \
       .select("s3_uri").collect()[0][-1]))},
  
  'LongTerm=':{'target': 'longterm_binary', 
                 'feature_key': 'longterm',
                'uri':  str((model_tracking_df \
       .withColumn('row', 
                  F.row_number().over(Window.partitionBy('model_name').orderBy(F.col("snapshot_date").desc()))) \
       .filter((F.col("model_name") == F.lit('LongTerm_model')) &
        (F.col("row") == F.lit(1))) \
       .select("s3_uri").collect()[0][-1]))},
  
  'AutoTransport':{'target': 'auto_transport_binary', 
                 'feature_key': 'auto_transport',
                'uri':  str((model_tracking_df \
       .withColumn('row', 
                  F.row_number().over(Window.partitionBy('model_name').orderBy(F.col("snapshot_date").desc()))) \
       .filter((F.col("model_name") == F.lit('autoTransport_model')) &
        (F.col("row") == F.lit(1))) \
       .select("s3_uri").collect()[0][-1]))},
   'checking':{'uri': 'ds.production_acorns_checking_full_base_deployed'},
  
  'Next_Positive':{'target': 'next_positive_binary', 
                 'feature_key': 'next_positive',
                'uri': str((model_tracking_df \
       .withColumn('row', 
                  F.row_number().over(Window.partitionBy('model_name').orderBy(F.col("snapshot_date").desc()))) \
       .filter((F.col("model_name") == F.lit('nextPositive_model')) &
        (F.col("row") == F.lit(1))) \
       .select("s3_uri").collect()[0][-1]))},
}
