import math
import numpy as np
import ast

def sampling(data, n_frames):
    fixed_data = []
    for i, record in enumerate(data):
        # used ast because the array being passed record is in the form of a string
        # For more info, try printing type(record) or print len(record)
        record = ast.literal_eval(record)
        center_of_list = math.floor(len(record)/2)
        crop_limit = math.floor(n_frames / 2)
        start = center_of_list - crop_limit
        end = center_of_list + crop_limit 
        # Add one more extra frame if n_frames is odd  
        paths_cropped = record[start: end + 1 if n_frames % 2 == 1 else end + 1]
        # Adding arrays of cropped clips for every video_sample
        fixed_data.append(paths_cropped)
    
    return fixed_data
