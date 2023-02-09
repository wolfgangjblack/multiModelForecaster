# Databricks notebook source
NOTEBOOK_NAME = "SL_utils"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import array, col, explode, struct, lit

# COMMAND ----------

def sl_utils_help():
  return """
  Loads in helper functions used in survivability layer to rank actions on a user level and suggest a sequence. 
  The SL returns a 100 if samples have ALREADY taken the action, otherwise it will suggest a sequence based on the action of
  the model prediction and the survivability metric. 
  
  Main function: get_sl_ranks
  """

# COMMAND ----------

st = spark.read.table('multimodel_forecaster.survivability_layer_table
                        ')

# COMMAND ----------

def to_long(df, by):
  """this takes a user and all of their action scores, pivots them so they have N rows corresponding to N actions. This allows us to rank those actions"""
  # Filter dtypes and split into column names and type description
  cols, dtypes = zip(*((c, t) for (c, t) in df.dtypes if c not in by))
  # Spark SQL supports only homogeneous columns
  assert len(set(dtypes)) == 1, "All columns have to be of the same type"

  # Create and explode an array of (column_name, column_value) structs
  kvs = explode(array([
    struct(lit(c).alias("key"), col(c).alias("score")) for c in cols
  ])).alias("kvs")

  return df.select(by + [kvs]).select(by + ["kvs.key", "kvs.score"])

# COMMAND ----------

def get_binaries_in_score_df(user_level_scores, users):
  """For essential actions, we use this to rank them either 0 or 1 on whether they've performed the base action. In V1, essential actions all samples are encouraged to take include buisness_requirement_1 and buisness_requirement_2"""
  binary_columns = [i for i in users.columns if 'binary' in i]
  return user_level_scores.join(users.select('uuid', *binary_columns), on ='uuid', how = 'left').fillna(0.0)

# COMMAND ----------

def get_scores(user_level_scores):
  """this returns the scores to be ranked, which is the product of the model prediction and the action survivability. If the sample has alread perfomed the action this is ranked as a -1, which later gets converted to a 100
  
  It should be noted here that if a model predicts that the user WILL NOT TAKE the action (prediction = 0) we instead return the action survivabilty*0.1. The reasoning behind this will be covered more in depth in the confluence, but basically we may have users who are predicted to take no action. In this instance, we still want to suggest to those users to take whatever action has the highest survivability, but we don't want those actions to supersede actions they may take"""
  user_ranks = user_level_scores \
       .withColumn('business_requirement_1_score',
                  F.when(F.col("business_requirement_1_binary") == F.lit(0),
                 F.col('business_requirement_1_binary')*F.lit(slt.select('business_requirement_1').collect()[0][-1]))
                  .otherwise(F.lit(-1))) \
      .withColumn('business_requirement_2_score',
                  F.when(F.col("business_requirement_2_binary") == F.lit(0),
                 F.col('business_requirement_2_binary')*F.lit(slt.select('business_requirement_2').collect()[0][-1]))
                  .otherwise(F.lit(-1)))\
      .withColumn('recurring_action_score',
                  F.when(F.col("recurring_action_binary") == F.lit(1),
                         F.lit(-1))
                    .when(F.col('recurring_action_prediction') == F.lit(0),
                          0.1*F.lit(slt.select('recurring_action_modifier').collect()[0][-1]))
                  .otherwise(F.col('recurring_action_prediction')*F.lit(slt.select('recurring_action_modifier').collect()[0][-1]))) \
      .withColumn('auto_transport_score',
                  F.when(F.col('auto_transport_binary') == F.lit(1),
                         F.lit(-1))
                  .when(F.col('auto_transport_prediction') == F.lit(0),
                        0.1*F.lit(slt.select('auto_transport_modifier').collect()[0][-1]))
                 .otherwise(F.col('auto_transport_prediction')*F.lit(slt.select('auto_transport_modifier').collect()[0][-1]))) \
      .withColumn('small_action_score',
                   F.when(F.col("small_action_binary") == F.lit(1),
                          F.lit(-1))
                  .when(F.col('small_action_prediction') == F.lit(0),
                        0.1*F.lit(slt.select('small_action_modifier').collect()[0][-1]))
                 .otherwise(F.col('small_action_prediction')*F.lit(slt.select('small_action_modifier').collect()[0][-1]))) \
      .withColumn('long_term_score',
                   F.when(F.col("long_term_binary") == F.lit(1),
                          F.lit(-1))
                  .when(F.col('long_term_prediction') == F.lit(0),
                        0.1*F.lit(slt.select('long_term_modifier').collect()[0][-1]))
                 .otherwise(F.col('long_term_prediction')*F.lit(slt.select('long_term_modifier').collect()[0][-1]))) \
      .withColumn('next_positive_score',
                F.when(((F.col("recurring_action_binary") == F.lit(1)) |
                       (F.col("next_positive_binary") == F.lit(1))), F.lit(-1))
                  .when(F.col('next_positive_prediction') == F.lit(0),
                        0.1*F.lit(slt.select('next_positive_modifier').collect()[0][-1]))
                 .otherwise(F.col('next_positive_prediction')*F.lit(slt.select('next_positive_modifier').collect()[0][-1]))) \
      .withColumn('transport_action_score',
                  F.when(F.col("transport_action_binary") == F.lit(1), F.lit(-1))
                  .when((F.col("auto_transport_prediction") == F.lit(1)),
                        (F.col('auto_transport_action_prediction')*F.lit(slt.select('auto_transport_modifier').collect()[0][-1]))-.0001)
                  .when(((F.col('auto_transport_prediction') == F.lit(0)) &
                        (F.col('transport_action_prediction') == F.lit(0))),
                          0.1*F.lit(slt.select('checking_modifier').collect()[0][-1]))
                  .when(((F.col('auto_transport_prediction') == F.lit(0)) &
                        (F.col('transport_action_prediction') == F.lit(1))),
                          F.lit(slt.select('transport_action_modifier').collect()[0][-1]))
                 )
  
  return user_ranks \
      .select('id',
              'business_requirement_1',
             'business_requirement_2',
             'recurring_action_modifier',
             'auto_transport_modifier',
             'next_positive_modifier',
             'small_action_modifier',
             'long_term_modifier',
             'transport_action_modifier')

# COMMAND ----------

def get_sl_ranks(user_level_scores, users):
  """this can be considered the main function, which returns the rank of each action at a user level. 
  If the user has performed the action, the action is given 100 to indicate that it has been performed"""
  rank_window = Window.partitionBy("id").orderBy(F.col('score').desc())
  user_level_scores = get_binaries_in_score_df(user_level_scores, users)
  user_ranks = get_scores(user_level_scores)
  return to_long(user_ranks, ["uuid"])\
          .withColumn("raw_rank",
                      F.rank().over(rank_window)) \
          .withColumn("rank",
                      F.when(F.col("score") == F.lit(-1), F.lit(100))
                      .otherwise(F.col("raw_rank"))) \
      .groupby("uuid") \
      .pivot("key") \
      .sum("rank")
