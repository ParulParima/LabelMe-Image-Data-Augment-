import numpy as np

class shape_adjustment:

    def __init__(self):
        pass

    # Adds shape_adjust or crops the background image accordingly to match dimensions of the annotated image                 
    def shape_adjust(self,res, bg_width, bg_height):        
        pad_w_tot = res.shape[1] - bg_width
        pad_h_tot = res.shape[0] - bg_height

        pad_bottom, pad_right = pad_h_tot, pad_w_tot

        if pad_w_tot<=0:
            pad_right = -pad_w_tot

        if pad_h_tot<=0:
            pad_bottom = -pad_h_tot

        # Same dimensions
        if pad_w_tot==0 and pad_h_tot==0:                   
            pass

        # Annotated Image dimensions (height, width) > Background Image dimensions
        elif pad_w_tot>0 and pad_h_tot>0:
            res = res[0:res.shape[0]-pad_bottom, 0:res.shape[1]-pad_right, :]

        # Annotated Image's width >= Background Image's width; Annotated Image's height <= Background Image's height
        elif pad_w_tot>=0 and pad_h_tot<=0:
            rb = np.pad(res[:,:,0],((0,pad_bottom),(0,0)), mode='constant', constant_values=0)
            gb = np.pad(res[:,:,1],((0,pad_bottom),(0,0)), mode='constant', constant_values=0)
            bb = np.pad(res[:,:,2],((0,pad_bottom),(0,0)), mode='constant', constant_values=0)       
            res = np.dstack(tup=(rb, gb, bb))
            res = res[:, 0:res.shape[1]-pad_right, :]

        # Annotated Image's width <= Background Image's width; Annotated Image's height >= Background Image's height
        elif pad_w_tot<=0 and pad_h_tot>=0:
            rb = np.pad(res[:,:,0],((0,0),(0,pad_right)), mode='constant', constant_values=0)
            gb = np.pad(res[:,:,1],((0,0),(0,pad_right)), mode='constant', constant_values=0)
            bb = np.pad(res[:,:,2],((0,0),(0,pad_right)), mode='constant', constant_values=0)
            res = np.dstack(tup=(rb, gb, bb))
            res = res[0:res.shape[0]-pad_bottom, :, :]

        # Annotated Image dimensions (height, width) < Background Image dimensions
        else:
            rb = np.pad(res[:,:,0],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
            gb = np.pad(res[:,:,1],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
            bb = np.pad(res[:,:,2],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
            res = np.dstack(tup=(rb, gb, bb)) 
    
        return res