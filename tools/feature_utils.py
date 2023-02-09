# Databricks notebook source
NOTEBOOK_NAME = "feature_utils"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

def feature_utils_help():
  return """NB Purpose:
This notebook contains the feature config necessary to generate features for all models within metamodel suite

How to use this:
  1. call this notebook to initialize functions. 
  2. generate_features function will read in a users table with eligible users (see above) and then utilize subfunctions
  
  Note: This code has been removed to protect the data
"""

