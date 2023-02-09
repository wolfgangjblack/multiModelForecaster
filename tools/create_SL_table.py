# Databricks notebook source
NOTEBOOK_NAME = "create_SL_table"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

# MAGIC %md
# MAGIC #notebook purpose:
# MAGIC 
# MAGIC In this notebook, we generate the survivability layer (SL) table used in the SL to help rank actions. Note, this table has modifiers based on a survivability metric described for internal business reasons. 
# MAGIC 
# MAGIC Note: a few of our actions are either abstracted or original. Each action is is listed below with its survivability and any notes. 
# MAGIC 
# MAGIC | action | survivability | notes| 
# MAGIC | - | - | - |
# MAGIC |'business_requirement_1'| 1.01 | This is not based on survivability, its a requirement from the business for each sample. If they haven't done this, this would be THE number 1 recommendation |
# MAGIC | 'business_requirement_2'| 1.00 | This is not based on survivability, its also a requirement from the business for each sample |
# MAGIC | 'recurring_action_modifier'| 0.9151 | TPV2 |
# MAGIC | 'auto_transport_modifier'| 0.6667 | This is not in the TPV2, it is slightly higher ranked than Checking |
# MAGIC | 'next_positive_modifier'| 0.8163 | This is the 3+ deposits/investments from TPV2 |
# MAGIC | 'small_action_modifier'| 0.7852 | TPV2 |
# MAGIC | 'long_term_modifier'| 0.6453| TPV2 |
# MAGIC | 'transport_action_modifier'| 0.6657| TPV2 |
# MAGIC 
# MAGIC <b>For any other future modeling efforts, this should be updated either to change the scorability as affected by the modeling, some post-modeling analytics, or to include any new models.</b>

# COMMAND ----------

from pyspark.sql import Row
import pyspark.sql.functions as F


##Add survivability here
modifiers =  [Row(1.01,
            1.00,
            0.9151,
            0.6667,
            0.81673,
           0.7852,
            0.6453,
            0.6657)]

rdd = spark.sparkContext.parallelize(modifiers)

##Add model name here
modifier_columns = ['business_requirement_1',
 'business_requirement_2',
 'recurring_action_modifier',
 'auto_transport_modifier',
 'next_positive_modifier',
 'small_action_modifier',
 'long_term_modifier',
 'transport_action_modifier']
df = rdd.toDF()

static_cols = df.columns

for i in range(len(static_cols)):
  df = df.withColumnRenamed(static_cols[i], modifier_columns[i])

# COMMAND ----------

df.display()

# COMMAND ----------

df.withColumn('snapshot_date',
                          F.lit(F.current_date())).write.mode("overwrite").option("overwriteSchema", True).saveAsTable("multimodel_forecaster.survivability_layer_table")
