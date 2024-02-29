import tkinter as tk
from tkinter import Button, Canvas
import numpy as np
import math
from rdp import rdp
from torch.optim import Adam

from demo_model import run_model
from display_data import display
from autoencoder import VAE
from data_processing import load_weights
from params import device

#Epsilon for rdp
scaling = 0.4

# Create main window
root = tk.Tk()
root.title("Draw with Cursor")

# Sets Size of Canvas, Creates Canvas Widget, Adds Canvas to the window
canvas_width, canvas_height = 500, 500
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.grid(row=0, column=0, sticky="nsew", columnspan=4)

# Variables to track last mouse position, List to store drawing data, Minimum distance between points to trigger a drawing action
last_x, last_y = None, None
drawing_data = []  
distance_threshold = 6
connector_line = True 

# Radio button variables
options = ["Apple", "Flower", "Cactus", "Carrot"]
sketch_class = tk.StringVar(value="Apple")

# Load model
model = VAE().to(device)
optimizer = Adam(model.parameters()) 
load_weights(model,optimizer,"model/final/remote/fruit.pt") 

model.generate = True

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
    if len(drawing_data) > 0:
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

def raise_popup():
    """
    Create a popup window if the drawing has too many strokes
    """

    # Create a popup window
    popup_window = tk.Toplevel(root)
    popup_window.geometry("250x100")

    # Create label
    message_label = tk.Label(popup_window, text="Drawing is too complex, please try again.")
    message_label.pack(pady=20)

    # Create Ok button
    ok_button = tk.Button(popup_window, text="Ok", command=popup_window.destroy)
    ok_button.pack()


# Save function
def process_image():
    global drawing_data  # Ensure we're using the global variable
    
    # Ensure drawing_data is a NumPy array for processing
    np_array = np.array(drawing_data, dtype=int)
    
    # Simplify the drawing data while keeping certain rows indicated by the third column
    if len(np_array) > 100:
        simplified_array = rdp_keep_rows(np_array, epsilon=scaling)
    else:
        simplified_array = np_array

    # Model can only handle up to 200 strokes
    if len(simplified_array) > 200:
        raise_popup()
    else:
        result = run_model(model, simplified_array, sketch_class.get(), mode='process')
        display(result)
    
    # Clear the drawing data and the canvas for new drawings
    drawing_data.clear()
    canvas.delete("all")

def complete_image():
    global drawing_data

    # Ensure drawing_data is a NumPy array for processing
    np_array = np.array(drawing_data, dtype=int)
    
    # Simplify the drawing data while keeping certain rows indicated by the third column
    if len(np_array) > 100:
        simplified_array = rdp_keep_rows(np_array, epsilon=scaling)
    else:
        simplified_array = np_array

    # Model can only handle up to 200 strokes
    if len(simplified_array) > 200:
        raise_popup()
    else:
        result = run_model(model, simplified_array, sketch_class.get(), mode='complete')
        display(result)

    # Clear the drawing data and the canvas for new drawings
    drawing_data.clear()
    canvas.delete("all")


# Creates a button that runs the save_image function when pressed
button_save = Button(root, text="Process Drawing", command=process_image)
# Adds the button to the window
button_save.grid(row=1, column=0, sticky="nsew", columnspan=2)

# Button for drawing completion
button_complete = Button(root, text="Complete Drawing", command=complete_image)
button_complete.grid(row=1, column=2, sticky="nsew", columnspan=2)

# Radio buttons for sketch class
for i, option in enumerate(options):
    rb = tk.Radiobutton(root, 
                        text=option, 
                        variable=sketch_class, 
                        value=option
                        )
    rb.grid(row=3, column=i, sticky="w")


# Starts the Tkinter event loop
root.mainloop()