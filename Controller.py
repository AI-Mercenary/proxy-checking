#controller
from Model import Model
from View import View
import os
from azure.storage.blob import BlobServiceClient

proxy_account_name="proxychecking"
proxy_account_key="MrArpUhcuURHmC4oCqAnxTCjOC1rJTa7f2DiNXHWmqh24E4mo1dHMPu8FqF4ISrGfMNARATw0q6w+AStW51h6g=="
proxy_container_name="screenshots"
proxy_connect_str = f"DefaultEndpointsProtocol=https;AccountName={proxy_account_name};AccountKey={proxy_account_key};EndpointSuffix=core.windows.net"

reaidy_account_name="reaidystorage"
reaidy_account_key="s5GikSOTgzRkzXQOMmwlmMeOn/5dlLocXGyccCR+Z10hWptcDWw8JsLl02pOYkprjbNfAOTgCRo9+AStKWz/2A=="
reaidy_container_name="recordings"
reaidy_connect_str = f"DefaultEndpointsProtocol=https;AccountName={reaidy_account_name};AccountKey={reaidy_account_key};EndpointSuffix=core.windows.net"

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
        self.proxy_account_name = proxy_account_name
        self.proxy_account_key = proxy_account_key
        self.proxy_container_name = proxy_container_name
        self.proxy_connect_str=proxy_connect_str
        self.proxy_blob_service_client = BlobServiceClient.from_connection_string(proxy_connect_str)
        self.proxy_container_client = self.proxy_blob_service_client.get_container_client(proxy_container_name)
        self.proxy_base_url = f"https://{self.proxy_account_name}.blob.core.windows.net/{self.proxy_container_name}/"
        
        self.reaidy_account_name = reaidy_account_name
        self.reaidy_account_key = reaidy_account_key
        self.reaidy_container_name = reaidy_container_name
        self.reaidy_connect_str=reaidy_connect_str
        self.reaidy_blob_service_client = BlobServiceClient.from_connection_string(reaidy_connect_str)
        self.reaidy_container_client = self.reaidy_blob_service_client.get_container_client(reaidy_container_name)
        self.reaidy_base_url = f"https://{self.reaidy_account_name}.blob.core.windows.net/{self.reaidy_container_name}/"


    def upload_to_blob_storage(self,file_path,file_name):
        blob_client =self.proxy_blob_service_client.get_blob_client(container=self.proxy_container_name, blob=file_name)        
        try:
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            msg= f"Uploaded and overwritten {file_name} file."
        except FileNotFoundError:
            msg= f"{file_name} not found. Check the file path."
        except Exception as e:
            msg= f"Error uploading {file_name} file."
        return msg
        
    
    def delete_blob_from_storage(self):
        screenshots=self.get_filenames_from_azure
        for file_name in screenshots:
            blob_client = self.proxy_blob_service_client.get_blob_client(container=self.proxy_container_name, blob=file_name)
            try:
                blob_client.delete_blob()
                msg=f"Deleted {file_name} file"
            except FileNotFoundError:
                msg=f"{file_name} not found. It may have already been deleted."
            except Exception as e:
                msg=f"Error deleting {file_name} file."
            print(msg)
            return msg
        
    def get_filenames_from_azure(self):
        return [blob.name for blob in self.proxy_container_client.list_blobs()]
    
    
    def get_video(self, file_name,source):
        if source=="reaidy":
            blob_client = self.reaidy_blob_service_client.get_blob_client(
                container=self.reaidy_container_name,
                blob=file_name
            )
        else:
            blob_client = self.proxy_blob_service_client.get_blob_client(
                container=self.proxy_container_name,
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
            urls.append(self.proxy_base_url + i)
        return urls

    
