# Databricks notebook source
NOTEBOOK_NAME = "feature_config"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

def feature_config_help():
  return """This notebook will create the feature_config file which will be used each model to determine the features. 

  This config will be a dictionary with model names as keys, and features as lists. 

  How this is used:

  1. features are generated by a feature generation script 
  2. specific features are passed to each model via a function call, using the key to identify which model will be used and the values to identify which features to pass in

  Note: The specifics of this has been removed to product the data
  """

# COMMAND ----------
