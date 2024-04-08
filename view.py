#view
class View:
    def __init__(self):
        self.timestamps={}
        
    def display(self,grp,faces,urls):
        self.timestamps["proxy"]=faces
        self.timestamps["urls"]=urls
        self.timestamps["grp"]=grp
        return self.timestamps