#controller
import cv2
import Model
import View
class Controller:
    def __init__(self,path):
        self.path = path
        self.output_path = "D:/sem 4-2/External Internship/project 5/User Analytics/proxy-checking/screenshots"
        self.model = Model(path)
        self.view = View()
        
    def check(self):
        grp,grp_count,faces,facing = self.model.checking_function()
        time_stamp = self.view.display(grp,grp_count,faces,facing)
        return time_stamp
    
    def image_screenshot(self,image,frame_count):
        image_screenshot = image.copy()
        screenshot_filename = f"frame_{frame_count}.jpg"
        screenshot_filepath = os.path.join(self.output_path, screenshot_filename)
        cv2.imwrite(screenshot_filepath, image_screenshot)
        return screenshot_filepath
                    
                    
    def frame2time(self,frame_count, frames_per_second=60):
        total_seconds = frame_count / frames_per_second
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        remaining_frames = int(frame_count % frames_per_second)
        return minutes, seconds, remaining_frames