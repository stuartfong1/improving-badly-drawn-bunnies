import numpy as np
import matplotlib.pyplot as plt
from encode_pen_state import encode_dataset1
from encoder_lstm import make_batch

# Get file path for one class of sketches
# data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
data_path = 'ambulance.npz'

# Load from file
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]

def display(imageNum):
    #Xplot and Yplot are array of points that will be plotted
    Xplot = [0]
    Yplot = [0]
    #Keeps track of the current point that is being drawn
    xpos = 0
    ypos = 0
    #While loop to go through data and plot points
    i=0
    while i < len(data[imageNum]):
        xpos += data[imageNum][i,0]
        ypos += data[imageNum][i,1]
        Xplot.append(-xpos)
        Yplot.append(-ypos)
        if data[imageNum][i,2] == 1:
            plt.plot(Xplot, Yplot,color='black')
            Xplot = []
            Yplot = []
        i+=1
#
# def display_encoded_image(image):
#     """
#     For some image tensor, draw the image using matplotlib.
#
#     Parameters:
#         - image: some [n*5] tensor representing a sketch.
#     Returns:
#         - none
#     """
#     #Xplot and Yplot are array of points that will be plotted
#     Xplot = [0]
#     Yplot = [0]
#     #Keeps track of the current point that is being drawn
#     xpos = 0
#     ypos = 0
#     #For loop to go through data and plot points
#     i=0
#     for i in range(len(image)):
#         xpos += image[i,0]
#         ypos += image[i,1]
#         Xplot.append(-xpos)
#         Yplot.append(-ypos)
#         if image[i,3] == 1:
#             plt.plot(Xplot, Yplot,color='black')
#             Xplot.clear()
#             Yplot.clear()
#         elif image[i, 4] == 1:
#             # plt.show()
#             break
#
#
# if __name__ == "__main__":
#     test_images, _ = make_batch(10)
#     encoded_images = np.array(test_images)
#     for i in range(10):
#         display_encoded_image(encoded_images[:,i])