# Databricks notebook source
NOTEBOOK_NAME = "train_models"
AUTHOR = "Wolfgang Black"

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook Purpose:
# MAGIC This notebook trains all the models in the current version of the multimodel forecaster suite. 
# MAGIC 
# MAGIC Please note:
# MAGIC 1. This is meant to be run on the HC cluster. 
# MAGIC 2. Each model takes approximately 5.5 hrs to run. 
# MAGIC 3. Simplicity here is intentional, we want to see each model output seperately
# MAGIC 4. To change model version or metrics, see ../tools/model_config and ../tools/model_utils

# COMMAND ----------

dbutils.widgets.text("model_train_list","recurring, small_action, auto_transport, longterm, next_positive")
model_train_list = dbutils.widgets.get("model_train_list").replace(' ', '').split(',')

# COMMAND ----------

model_train_list

# COMMAND ----------

if 'recurring' in model_train_list:
   dbutils.notebook.run("./autoinvest_development/recurring_model_dev", 22000)

# COMMAND ----------

if 'longterm' in model_train_list:
   dbutils.notebook.run("./later_development/later_model_dev", 22000)

# COMMAND ----------

if 'small_action' in model_train_list:
   dbutils.notebook.run("./roundup_development/small_action_model_dev", 22000)

# COMMAND ----------

if 'auto_transport' in model_train_list:
   dbutils.notebook.run("./direct_deposit_development/auto_transport_model_dev", 22000)

# COMMAND ----------

if 'next_positive' in model_train_list:
   dbutils.notebook.run("./direct_deposit_development/next_positive_model_dev", 22000)

# COMMAND ----------

dbutils.notebook.exit('')
