import tkinter as tk
from tkinter import Button, Canvas
import numpy as np
import math
from rdp import rdp

from demo_model import run_model
from display_data import display

#Epsilon for rdp
scaling = 0.4

# Create main window
root = tk.Tk()
root.title("Draw with Cursor")

# Sets Size of Canvas, Creates Canvas Widget, Adds Canvas to the window
canvas_width, canvas_height = 500, 500
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Variables to track last mouse position, List to store drawing data, Minimum distance between points to trigger a drawing action
last_x, last_y = None, None
drawing_data = []  
distance_threshold = 6
connector_line = True 

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Draw on canvas function
def paint(event):
    global last_x, last_y, connector_line
    # Checks if there is a previous point to draw from
    if last_x is not None and last_y is not None:
        # Check if distance is above threshold
        if calculate_distance(last_x, last_y, event.x, event.y) >= distance_threshold:
            if connector_line:
                canvas.create_line(last_x, last_y, event.x, event.y, fill='black', width=5)
            else:
                connector_line = True
            drawing_data.append([event.x - last_x, event.y - last_y, 0])
            last_x, last_y = event.x, event.y
    else:
        # Initalizes last point to current point if last point is none
        last_x, last_y = event.x, event.y

# Binds the paint function to the mouse drag events with the left button
canvas.bind('<B1-Motion>', paint)


def reset(event):
    global last_x, last_y, connector_line
    # Indicate the pen lift without recording a movement
    drawing_data[-1][2] = 1
    # drawing_data.append([0, 0, 1])
    connector_line = False


# Binds the reset function to left mouse button release events
canvas.bind('<ButtonRelease-1>', reset)

def rdp_keep_rows(array, epsilon=scaling, keep_column=2):
    # Extract the 2D slice and rows to keep
    two_d_slice = array[:, :2]
    keep_rows = array[:, keep_column] == 1

    # Apply RDP
    simplified = rdp(two_d_slice, epsilon=epsilon)

    # Convert simplified to a set of tuples for fast lookup
    simplified_set = set(map(tuple, simplified))

    # Prepare the final list including rows to keep
    final_list = []
    for idx, point in enumerate(two_d_slice):
        if tuple(point) in simplified_set or keep_rows[idx]:
            # Reconstruct the original 3D point with the third value as zero or one
            final_list.append(list(point) + [int(keep_rows[idx])])

    # Convert list to a 3D array
    final_3d = np.array(final_list, dtype=array.dtype)
    
    return final_3d


# Save function
def process_image():
    global drawing_data  # Ensure we're using the global variable
    filename = "python/arrayStuff.py"
    
    # Ensure drawing_data is a NumPy array for processing
    np_array = np.array(drawing_data, dtype=int)
    
    # Simplify the drawing data while keeping certain rows indicated by the third column
    if len(np_array) > 100:
        simplified_array = rdp_keep_rows(np_array, epsilon=scaling)
    else:
        simplified_array = np_array

    result = run_model(simplified_array)
    display(result)


    
    # # Format the simplified NumPy array to string for saving
    # formatted_output = '[' + ',\n'.join(['[' + ', '.join(map(str, row)) + ']' for row in simplified_array]) + ']'
    # formatted_output = "import numpy as np\nmy_array = np.array(" + formatted_output + ", dtype=np.int16)"
    
    # # Write the formatted string to the file
    # with open(filename, 'w') as file:
    #     file.write(formatted_output)
    
    # print(f"Drawing data saved as {filename}")
    
    # Clear the drawing data and the canvas for new drawings
    drawing_data.clear()
    canvas.delete("all")

# Creates a button that runs the save_image function when pressed
button_save = Button(root, text="Save Drawing", command=process_image)
# Adds the button to the window
button_save.pack()

# Starts the Tkinter event loop
root.mainloop()