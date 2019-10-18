### Replicating paper results
Unfortunately, one cannot include the exact data utilized to the train both the clinical models due to HIPPA constraints.
The data can be found here if you fill out the appropriate agreements:
https://portal.dbmi.hms.harvard.edu/data-challenges/

To replicate results in paper, please see the folder: [/examples/ml4health_2019_replication](/examples/ml4health_2019_replication).
This will simply requiring copying the downloaded data into the appropriate directory and running scripts.


#### Training on new datasets
For training, simply alter the config.ini present in /examples file for your purposes. Relevant variables are:

- model_storage_directory: directory to store logging information, tensorboard checkpoints, model checkpoints
- bert_model_path: the file path to a pretrained bert model. Can be the pytorch-transformers alias.
- labels: an ordered list of labels you are training against. this should match the order given in a .fit() instance.