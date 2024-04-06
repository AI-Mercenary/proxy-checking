from Controller import Controller


def main():
    video_path="H:/proxy/test.mp4"
    controller=Controller(video_path)
    timestamp=controller.check()
    print(timestamp)
    
    
if __name__=="__main__":
    main()