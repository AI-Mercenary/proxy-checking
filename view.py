#view
class View:
    def __init__(self):
        self.timestamps={}
        
    def display(self,grp,faces,urls,msg,timings):
        self.timestamps["proxy"]=faces
        self.timestamps["urls"]=urls
        self.timestamps["grp"]=grp
        self.timestamps["Number of Speakers"]=msg
        if msg=="Multiple":
            self.timestamps["timings of disturbance"]=timings
        
        return self.timestamps