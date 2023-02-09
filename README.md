# MultiModel Forecaster
author: Wolfgang Black <br>
period: <br>
version: 1.0 <br>

## How to use this:

### Deployment:
This was developed in a DataBricks notebook environment. To deploy, call the /deployment/deploy.py. This produces the [user level scores]() and the [user level ranks](), which are both are saved to s3. The former is raw individual model output, but the latter is the rank ordered score, or the individual model prediction multiplied by the survival metric.

### Updating Meta-Model
To update the full meta-model, one should pay attention to model_config, feature_config, and create_TPDL_table notebooks. In these, models need to be manually added with their necessary components (model uri, feature set, and survivability). 

## Summary
The multiModelForecaster is a meta-model suite which predict whether a sample will perform certain action. Samples which score high in propensity to perform actions are shown the actions which are prioritized according to an obscured survivability metric. As this is the MVP, currently we push the action with the HIGHEST immediate increase in survivability metric, as opposed to a gradual increase in the metric. This meta-model suite currently contains models to predict propensity for the following actions:

 - Next Positive Propensity
 - Recurring Action Propensity
 - Small Action Propensity
 - Transport Action Propensity
 - Auto-Transport Propensity
 - Longterm Surviving Action Propensity

there are also two binary models based on business decisions, which if a sample hasn't taken that specific action are recommended as the number 1 and number 2 actions.

### Note:
 1. Actions here are fully anonymized, however code structure is provided for future adaptability and reproducability
 2. The Transport Action model was developed external to this project, and so it will be pulled down from an s3 location and utilized within the /deployment/deploy.py script
