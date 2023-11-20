import numpy as np

def encode_pen_state(sketch):
    """
    One-hot encode pen state by adding additional columns for pen up and end of stroke.
    
    Parameters: 
        sketch (ndarray): n*3 array with format (x,y,p1), representing sketch data
        
    Returns: 
        ndarray: n*5 array with format (x,y,p1,p2,p3), where p2 = 1-p1 and p3 is 1 at 
        end of the sketch, 0 otherwise.
    """
    
    shape = sketch.shape
    pen_up = (np.ones(shape[0]) - sketch[:,2]).reshape(shape[0],1)
    end_stroke = np.zeros((shape[0],1))
    end_stroke[-1] = 1 
    sketch[-1][2] = 0
    
    return np.concatenate((sketch,pen_up,end_stroke),axis=1)


def encode_dataset1(data):
    """
    Encode pen states by creating a new array of sketch data.
    
    Parameters:
        data (iterable): object containing data for each sketch
        
    Returns:
        ndarray: object array containing encoded data for each sketch
    """
    new_data = np.empty(data.size,dtype=object)

    for i, sketch in enumerate(data):
        new_data[i] = encode_pen_state(sketch) 

    return new_data


def encode_dataset2(data):
    """
    Encode pen states by modifying original dataset.
    
    Parameters:
        data (iterable): object containing data for each sketch
        
    Returns:
        None
    """
    for i, sketch in enumerate(data):
        data[i] = encode_pen_state(sketch) 
    return

