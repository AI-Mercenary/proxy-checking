from Controller import Controller,Database


def upload_video():
    database=Database()
    video_path="C:/Users/harsh/OneDrive/Desktop/WhatsApp Video 2024-04-08 at 16.33.56_6fb5770d.mp4"
    file_name="testfile asraf"
    database.upload_to_blob_storage(video_path,file_name) 
    return file_name 
      
def delete_files():
        database=Database()
        database.delete_blob_from_storage()
        
def main():
    local=True
    controller=Controller()
    database=Database()
    if local:
        file_name=upload_video()
        filepath=database.get_video(file_name,"local")
    else:
        filename="6603e300f1a59bbcdc0d711e.webm"
        filepath=database.get_video(filename,"reaidy")
    timestamp=controller.check(filepath)
    for key in timestamp:
        print(key,timestamp[key])
    
    
if __name__=="__main__":
    main()
    