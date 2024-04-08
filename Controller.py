#controller
from Model import Model
from View import View
import os
from azure.storage.blob import BlobServiceClient

account_name="proxychecking"
account_key="MrArpUhcuURHmC4oCqAnxTCjOC1rJTa7f2DiNXHWmqh24E4mo1dHMPu8FqF4ISrGfMNARATw0q6w+AStW51h6g=="
container_name="screenshots"
connect_str = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

class Controller:
    def __init__(self):
        self.database=Database()
        self.model = Model()
        self.view = View()
        
    def check(self,filepath):
        grp,faces = self.model.cheking_function(filepath,self.database)
        urls=self.database.get_image_url()
        time_stamp = self.view.display(grp,faces,urls)
        return time_stamp
    

class Database:
    def __init__(self):
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.connect_str=connect_str
        self.blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.base_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/"


    def upload_to_blob_storage(self,file_path,file_name):
        blob_client =self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)        
        try:
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            msg= f"Uploaded and overwritten {file_name} file."
        except FileNotFoundError:
            msg= f"{file_name} not found. Check the file path."
        except Exception as e:
            msg= f"Error uploading {file_name} file."
        return msg
        
    
    def delete_blob_from_storage(self, file_name):
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)

        try:
            blob_client.delete_blob()
            msg=f"Deleted {file_name} file"
        except FileNotFoundError:
            msg=f"{file_name} not found. It may have already been deleted."
        except Exception as e:
            msg=f"Error deleting {file_name} file."
        return msg
        
    def get_filenames_from_azure(self):
        return [blob.name for blob in self.container_client.list_blobs()]
    
    
    def get_video(self, file_name):
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=file_name
        )

        with open(file_name, "wb") as f:
            download_stream = blob_client.download_blob()
            download_stream.readinto(f)

        print(f"Video '{file_name}' downloaded successfully.")
        return file_name
    
    def get_image_url(self):
        urls=[]
        screenshots=self.get_filenames_from_azure()
        for i in screenshots:
            urls.append(self.base_url + i)
        return urls

    
