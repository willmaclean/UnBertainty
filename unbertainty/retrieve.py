import sys
import os
import boto3

"""
    Retrieves reports from S3 bucket.

    argv[1] = bucket name
    argv[2] = s3 folder
"""
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

if __name__ == '__main__':

    bucket_name = sys.argv[1]
    s3_folder = sys.argv[2]

    #Retrieve
    download_s3_folder(bucket_name, s3_folder)
