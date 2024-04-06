#controller
import cv2
from Model import Model
from View import View
import os

class Controller:
    def __init__(self,path):
        self.path = path
        self.model = Model(path)
        self.view = View()
        
    def check(self):
        grp,grp_count,faces,facing = self.model.cheking_function()
        time_stamp = self.view.display(grp,grp_count,faces,facing)
        return time_stamp
    
