Data cannot be directly included with source code.
You must sign the appropriate agreements [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

After clone this repository, place the datasets here. The data is spread over multiple files for each dataset, find
below a step-by-step guide to insure you have the appropriate data.
#### N2C2 2006
From "2006 De-identification and Smoking Status Challenge Downloads" section on DBMI portal:

In this directory (`examples/ml4health_2019_replication/data`), un-zip the files:
`smokers_surrogate_test_all_version2.zip`

`smokers_surrogate_train_all_version2.zip`.

This should yield two XML files:

`smokers_surrogate_test_all_version2.xml`

`smokers_surrogate_train_all_version2.xml`.

#### N2C2 2008
From "2008 Obesity Challenge Downloads" section on DBMI portal you want the XML files:

1. Training

`obesity_patient_records_training.xml`

`obesity_patient_records_training2.xml`

`obesity_standoff_annotations_training.xml`

`obesity_standoff_annotations_training_addendum.xml`

`obesity_standoff_annotations_training_addendum2.xml`

`obesity_standoff_annotations_training_addendum3.xml`


2.
`obesity_patient_records_test.xml`

`obesity_standoff_annotations_test.xml`

These are located in the zip files in the section. Pull them down and un-zip everything into this directory (again `examples/ml4health_2019_replication/data`)


As a pre-flight check, your data directory should look like this.

```bash
examples/ml4health_2019_replication/data/
├── obesity_patient_records_test.xml
├── obesity_patient_records_training2.xml
├── obesity_patient_records_training.xml
├── obesity_standoff_annotations_test.xml
├── obesity_standoff_annotations_training_addendum2.xml
├── obesity_standoff_annotations_training_addendum3.xml
├── obesity_standoff_annotations_training_addendum.xml
├── obesity_standoff_annotations_training.xml
├── README.md
├── smokers_surrogate_test_all_version2.xml
└── smokers_surrogate_train_all_version2.xml

```

The .gitignore will automatically ignore pushing up any XML files in this directly for PHI protection.
