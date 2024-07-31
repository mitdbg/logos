## `dataset_files/`

This directory holds the following types of files, for each of the datasets `x` used in our evaluation:
- `x/datasets_raw/`: The raw logs.
- `x/datasets/`: The cached byproducts of processing the dataset with LogOS, in pickled form.
- `x/evaluation/`: The outputs produced by our experiment runners when processing the dataset in question.

Some of these files are large, which is why we have hosted them on S3 instead of distributing them
inside this repository. If you would like to access any of these datasets, please email us at markakis[at]mit[dot]edu.

Once you have been granted access, you can download the `PostgreSQL` dataset by running:
```sh
aws s3 sync s3://logos-dataset-postgresql postgresql/
```

Once you have been granted access, you can download the `XYZ` dataset by running:
```sh
aws s3 sync s3://logos-dataset-xyz xyz/
```

Once you have been granted access, you can download the datasets for the scaling microexperiments by running:
```sh
aws s3 sync s3://logos-dataset-scaling scaling/
```

The `Proprietary` dataset is not publicly available for privacy reasons. If you have an extremely compelling reason to request access, please explain it when requesting access and we may review your request on a case-by-case basis. If you have been granted access, you can download the `Proprietary` dataset by running:
```sh
aws s3 sync s3://logos-dataset-proprietary proprietary/
```