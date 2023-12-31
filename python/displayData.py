import numpy as np
import matplotlib.pyplot as plt

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