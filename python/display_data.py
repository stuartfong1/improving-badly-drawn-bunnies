import numpy as np
import matplotlib.pyplot as plt

def display(my_array):
    my_array = np.array(my_array.cpu())
    #Xplot and Yplot are array of points that will be plotted
    Xplot = []
    Yplot = []
    #Keeps track of the current point that is being drawn
    xpos = 0
    ypos = 0
    #While loop to go through data and plot points
    i=0
    while i < len(my_array):    
        xpos += my_array[i,0]
        ypos += my_array[i,1]
        Xplot.append(xpos)
        Yplot.append(-ypos)
        if my_array[i,2] == 1:
            plt.plot(Xplot, Yplot,color='black', linewidth=3)
            Xplot = []
            Yplot = []
        i+=1
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.show()
    