#view
class View:
    def __init__(self,grp,grp_count,faces,facing):
        self.grp=grp
        self.grp_count=grp_count
        self.faces=faces
        self.facing=facing
        
    def display(self):
        timestamps={}
        timestamps["camera_facing"]=self.facing
        timestamps["multiple_faces"]=self.faces
        timestamps["grp"]=self.grps
        return timestamps