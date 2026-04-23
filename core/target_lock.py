class TargetLock:

    def __init__(self,width,height):

        self.center_x = width//2
        self.center_y = height//2

        self.prev_x = self.center_x
        self.prev_y = self.center_y

        self.smooth = 0.2

    def compute(self,bbox):

        x1,y1,x2,y2 = bbox

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        err_x = cx - self.center_x
        err_y = cy - self.center_y

        move_x = int(self.prev_x + self.smooth*err_x)
        move_y = int(self.prev_y + self.smooth*err_y)

        self.prev_x = move_x
        self.prev_y = move_y

        return move_x,move_y,cx,cy