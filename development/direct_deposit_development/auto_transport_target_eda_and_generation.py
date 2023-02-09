# Databricks notebook source
NOTEBOOK_NAME = "auto_transport_target_eda_and_generation"
AUTHOR = "Wolfgang Black"
DATE_MODIFIED = "2022-09-13"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Purpose:
# MAGIC 
# MAGIC In this notebook we'll grab the user population we want to train on. The steps are shown below:
# MAGIC 1. Get users who are verified this year and still open
# MAGIC 2. using the tools/feature_utils nb, get all features
# MAGIC 3. using the tools/feature_config nb, get features and binary for specific model
# MAGIC 4. get distribution for users with and without target
# MAGIC 6. sample down to the minority class numbers to ensure a class balance for modeling
# MAGIC 7. check distributions between downsample majority class and all majority class - if similiar distributions, save dataframe for training and validation
# MAGIC 
# MAGIC <b> Notes </b>
# MAGIC 1. targets are consider only for some fixed time, and not all time.
# MAGIC     - this means for an action_binary = 1, the sample did not take the action within specified time periond
# MAGIC     - similarly, action_binary = 0 means a user may have taken AFTER the specified time periond
# MAGIC 
# MAGIC To do:
# MAGIC 1. we already have our users and features
# MAGIC 2. we need to get a dataset for users who have and haven't taken a specific action
# MAGIC 4. once we have this, we save users 

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

## Takne in specific tables

# COMMAND ----------

# DBTITLE 1,get eligible users
##users = some filtering of above tables by time period 

# COMMAND ----------

# DBTITLE 1,Get generate_features func
# MAGIC %run ../../tools/feature_utils

# COMMAND ----------

# DBTITLE 1,get all features
users = generate_features(users, 'some_time_periond')

# COMMAND ----------

# DBTITLE 1,Get feature_config to see what features are used in this (later) model
# MAGIC %run ../../tools/feature_config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get features for this model

# COMMAND ----------

features = feature_config['auto_transport']

# COMMAND ----------

features

# COMMAND ----------

users = users.select("id", *features, 'auto_transport_binary')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get positive/negative groups for training

# COMMAND ----------

users.groupby('auto_transport_binary').agg(F.count(F.lit(1))).display()

# COMMAND ----------

from pyspark.sql.functions import isnull, when, count, col
users.select([count(when(isnull(c), c)).alias(c) for c in users.columns]).display()

# COMMAND ----------

users = users.replace('null', None)\
        .dropna(how='any')
users.select([count(when(isnull(c), c)).alias(c) for c in users.columns]).display()

# COMMAND ----------

users.count()

# COMMAND ----------

positive_class = users.filter(F.col("auto_transport_binary") == F.lit(1))
negative_class_all = users.filter(F.col("auto_transport_binary") == F.lit(0))
target_percent = positive_class.count()/negative_class_all.count()
print(positive_class.count(), negative_class_all.count(),target_percent)

# COMMAND ----------

negative_class = negative_class_all.sample(fraction = target_percent, seed = 123)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get distributions per df

# COMMAND ----------

[get_barplot(users,users.columns[i],'') for i in range(len(users.columns)) if ((i > 0) & (i < 21))]

# COMMAND ----------

[get_barplot(negative_class_all,negative_class_all.columns[i],'') for i in range(len(negative_class_all.columns)) if ((i > 0) & (i < 21))]

# COMMAND ----------

[get_barplot(negative_class,negative_class.columns[i],'') for i in range(len(negative_class.columns)) if ((i > 0) & (i < 21))]

# COMMAND ----------

[get_barplot(positive_class,positive_class.columns[i],'') for i in range(len(positive_class.columns)) if ((i > 0) & (i < 21))]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save off data
# MAGIC 
# MAGIC Note: to save data, we need to change the names back to the axciom names

# COMMAND ----------

get_acxiom_codes(users) \
     .withColumn('snapshot_date',
                          F.lit(F.current_date())) \
      .write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.auto_transport_users_full_population")

get_acxiom_codes(positive_class).withColumn('snapshot_date',
                          F.lit(F.current_date())).write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.auto_transport_users_positive_class")

get_acxiom_codes(negative_class).withColumn('snapshot_date',
                          F.lit(F.current_date())).write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.auto_transport_users_negative_class")

get_acxiom_codes(negative_class_all).withColumn('snapshot_date',
                          F.lit(F.current_date())).write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.auto_transport_users_negative_class_all")

# COMMAND ----------
