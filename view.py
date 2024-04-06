#view
class View:
    def __init__(self):
        self.timestamps={}
        
    def display(self,grp,grp_count,faces,facing):
        self.timestamps["camera_facing"]=facing
        self.timestamps["multiple_faces"]=faces
        self.timestamps["grp"]=grp
        return self.timestamps