import tkinter as tk
from tkinter import Button, Canvas
import numpy as np
import math

# Create main window
root = tk.Tk()
root.title("Draw with Cursor")

# Create a canvas to draw on
canvas_width, canvas_height = 400, 400
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Variables to track drawing
last_x, last_y = None, None
drawing_data = []  # Keep using a list for dynamic data collection
distance_threshold = 5  # Minimum distance between points

# Function to calculate distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Draw function
def paint(event):
    global last_x, last_y
    if last_x is not None and last_y is not None:
        # Check if distance is above threshold
        if calculate_distance(last_x, last_y, event.x, event.y) >= distance_threshold:
            canvas.create_line(last_x, last_y, event.x, event.y, fill='black', width=5)
            drawing_data.append([event.x - last_x, event.y - last_y, 0])
            last_x, last_y = event.x, event.y
    else:
        last_x, last_y = event.x, event.y

canvas.bind('<B1-Motion>', paint)

# Reset last_x and last_y when the pen is lifted
def reset(event):
    global last_x, last_y
    if last_x is not None and last_y is not None:
        drawing_data.append([0, 0, 1])
    last_x, last_y = None, None

canvas.bind('<ButtonRelease-1>', reset)

# Save function
def save_image():
    filename = "drawing_data.txt"
    np_array = np.array(drawing_data, dtype=int)  # Convert to NumPy array
    
    # Format the NumPy
    array_str = np.array2string(np_array, separator=', ')
    array_str = "array(" + array_str + ", dtype=int16)"
    
    # Write to the file
    with open(filename, 'w') as file:
        file.write(array_str)
    
    print(f"Drawing data saved as {filename}")
    drawing_data.clear()  # Clear the drawing data after saving
    canvas.delete("all")  # Clear the canvas

button_save = Button(root, text="Save Drawing", command=save_image)
button_save.pack()

root.mainloop()
