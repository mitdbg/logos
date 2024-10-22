# For internal use only - you won't have permissions for this.
aws s3 sync postgresql/ s3://logos-dataset-postgresql --delete --exclude "datasets_raw/*" --exclude "repro_evaluation/*"
aws s3 sync proprietary/ s3://logos-dataset-proprietary --delete --exclude "datasets_raw/*" --exclude "repro_evaluation/*"
aws s3 sync xyz/ s3://logos-dataset-xyz --delete --exclude "datasets_raw/*" --exclude "repro_evaluation/*"
aws s3 sync scaling/ s3://logos-dataset-scaling --delete --exclude "datasets_raw/*" --exclude "repro_evaluation/*"
