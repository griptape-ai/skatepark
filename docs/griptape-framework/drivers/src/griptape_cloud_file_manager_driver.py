import os

from griptape.drivers import GriptapeCloudFileManagerDriver

gtc_file_manager_driver = GriptapeCloudFileManagerDriver(
    api_key=os.environ["GT_CLOUD_API_KEY"],
    bucket_id=os.environ["GT_CLOUD_BUCKET_ID"],
)

# Download File
file_contents = gtc_file_manager_driver.try_load_file(os.environ["GT_CLOUD_ASSET_NAME"])

print(file_contents.decode())

# Upload File
response = gtc_file_manager_driver.try_save_file(os.environ["GT_CLOUD_ASSET_NAME"], file_contents)

print(response)
