aws s3 sync s3://logos-dataset-postgresql postgresql/  --delete --exclude "repro_evaluation/*"
aws s3 sync s3://logos-dataset-proprietary proprietary/ --delete --exclude "repro_evaluation/*"
aws s3 sync s3://logos-dataset-xyz xyz/ --delete --exclude "repro_evaluation/*"
aws s3 sync s3://logos-dataset-scaling scaling/ --delete 
