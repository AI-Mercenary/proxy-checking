from Controller import Controller,Database


def upload_video():
    database=Database()
    video_path="H:/proxy/test.mp4"
    file_name="testfile"
    database.upload_to_blob_storage(video_path,file_name) 
    return file_name 
      
      
      
def main():
    filename=upload_video()
    controller=Controller()
    database=Database()
    filepath=database.get_video(filename)
    timestamp=controller.check(filepath)
    for key in timestamp:
        print(key,timestamp[key])
    
    
if __name__=="__main__":
    main()
    