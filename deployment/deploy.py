# Databricks notebook source
NOTEBOOK_NAME = "deploy"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

# MAGIC %md
# MAGIC #NB Purpose:
# MAGIC 
# MAGIC In this notebook the following occurs
# MAGIC 1. we call the tables necessary to generate the user
# MAGIC 2. using <b>../tools/feature_config</b> and <b>../tools/feature_utils</b> we generate features for the mulitmodel forecaster models
# MAGIC 3. using <b>../tools/model_utils</b> and <b>../tools/model_config</b> we generate scores for each sample and return the raw probability scores and prediction to do an action per sample
# MAGIC 4. using <b>../tools/SL_utils</b> we rank the actions from 1-N, where N is the number of models/decisions considered for each sample
# MAGIC 5. using <b>../tools/model_utils/save_inference</b> we save the final table to <b>'multimodel_forecaster.user_level_suggestions'</b>

# COMMAND ----------

import pyspark.sql.functions as F
import datetime

# COMMAND ----------

# DBTITLE 1,load in feature_config - here we get lists of feature sets per model
# MAGIC %run ../tools/feature_config

# COMMAND ----------

# DBTITLE 1,load in feature_utils. This loads in base tables and functions necessary to calc all features. Main func is generate_features
# MAGIC %run
# MAGIC ../tools/feature_utils

# COMMAND ----------

# DBTITLE 1,load in model_config. Calls model_tracking_df which saves the uri's of all models. Will use latest models for inference
# MAGIC %run
# MAGIC ../tools/model_config

# COMMAND ----------

# DBTITLE 1,load in model_utils. inits functions for model inference
# MAGIC %run
# MAGIC ../tools/model_utils

# COMMAND ----------

# DBTITLE 1,load in SL_utils. inits functions for score ranking 
# MAGIC %run
# MAGIC ../tools/SL_utils

# COMMAND ----------

# DBTITLE 1,Eligible Users 
time_period = (datetime.datetime.today()-datetime.timedelta(int('some_days_prior')).strftime("%Y-%m-%d") #Note: this has been obscured, but should be changed if code is reproduced
users = spark.read.table("raw_users") \
    .select("uuid", "id", 'verified_at', 'closed_at')

# COMMAND ----------

# DBTITLE 1,Get all features for all models
users = generate_features(users)

# COMMAND ----------

# MAGIC %md
# MAGIC We have no models for users to do business_reasons 1 & 2. However, we've determined these to be necessary steps for a sample to have a long survivability. 
# Magic   As such, we'll start our user_level_scores table with these two binaries.

# COMMAND ----------

# DBTITLE 1,Get users and 'score' them for binaries
user_level_scores = get_binary_decisioning(users)

# COMMAND ----------

# DBTITLE 1,Score all users across all models in model_run_config
for model_key in model_run_config.keys():
  print(model_key)
    
  if model_key == 'transport':
    ##Note: checking inference is done seperately monthly (see confluence)
    user_level_scores = user_level_scores \
      .join((get_model_score_deciles(get_transport_score(users \
            .filter(F.col("transport_binary") == F.lit(0)) \
            .select("uuid")), model_key)
            .union(users \
                   .filter(F.col("transport_binary") == F.lit(1)) \
                   .select('uuid') \
                   .withColumn('transport_prediction', F.lit(0.0))
                   .withColumn('transport_probability', F.lit(0.0))
                   .withColumn('transport_score_decile', F.lit(0)))),
            on = 'uuid', how = 'left')
    
  else:
    print("getting user populations")
    users_to_score = get_acxiom_codes(users \
      .filter(F.col(model_run_config[model_key]['target']) == F.lit(0)) \
      .select("uuid", *feature_config[model_run_config[model_key]['feature_key']]) \
      .replace('null', None) \
        .dropna(how='any'))
    
    users_unscored = users.join(users_to_score, on = 'uuid', how = 'leftanti').select('uuid')
    
    print('scoring users')
    users_scored = load_model_and_evaluate(users_to_score, model_key, model_run_config[model_key]['uri'])
    
    user_level_scores = user_level_scores \
        .join(get_model_score_deciles(users_scored, model_key).union(users_unscored \
              .select("uuid") \
              .withColumn(model_key+'_probability', F.lit(0.0)) \
              .withColumn(model_key+"_prediction", F.lit(0.0)) \
              .withColumn(model_key+"_score_decile", F.lit(0))), on = 'uuid', how = 'left').fillna(0.0)

# COMMAND ----------

user_level_scores.summary().display()

# COMMAND ----------

# DBTITLE 1,Save (overwrite) user scores
user_level_scores.withColumn('snapshot_date',
                          F.lit(F.current_date())).write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.user_level_scores")

# COMMAND ----------

# DBTITLE 1,Get ranks per action
user_level_ranks = get_sl_ranks(user_level_scores, users)

# COMMAND ----------

user_level_ranks.summary().display()

# COMMAND ----------

# DBTITLE 1,Save (overwrite) user action ranks
user_level_ranks.withColumn('snapshot_date',
                          F.lit(F.current_date())).write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.user_level_ranks")

# COMMAND ----------
# in production, this has a connection to a stakeholder microservice whichs allows internal stakeholders to call down samples and action probability for business use.
# This was removed from this code for privacy reasons
