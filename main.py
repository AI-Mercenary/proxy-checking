from Controller import Controller,Database
import re
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
# Restrict TensorFlow to use only CPU devices
tf.config.experimental.set_visible_devices([], 'GPU')
      
def delete_files():
        database=Database()
        database.delete_blob_from_storage()
       
def main():
    local=False
    controller=Controller()
    database=Database()
    if local:
        video_path="H:/proxy/videos/interview.mp4"
        match = re.search(r'[^/]+\.mp4$', video_path)
        file_name = match.group(0)
        database.upload_to_blob_storage(video_path,file_name) 
        filepath=database.get_video(file_name,"local")
    else:
        filename="65fe63139ec3e5b8f2de262b.webm"
        filepath=database.get_video(filename,"reaidy")
    timestamp=controller.check(filepath)
    for key in timestamp:
        print(key,timestamp[key])
        
    
if __name__=="__main__":
    main()
    