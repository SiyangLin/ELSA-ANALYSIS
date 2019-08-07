# -*- coding: utf-8 -*-
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

def plot3D(X, y, title, savePath):
    fig = plt.figure(1, figsize=(6.5,5))
    ax = Axes3D(fig, elev=50, azim=50)
    scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc='lower left', title='health')
    ax.add_artist(legend1)
    
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_zlabel('dim3')
    ax.set_title(title)
    fig.savefig(savePath)    

def plot2D(X, y, title, savePath):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:,0], X[:,1], c=y)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc='lower left', title='health')
    ax.add_artist(legend1)
    
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_title(title)
    fig.savefig(savePath)

if __name__=='__main__':
    pca_3d = PCA(n_components=2)
    Data_3d = pca_3d.fit_transform(df)
    
    plot2D(Data_3d, label.to_numpy(), 'PCA 2-d','2d.png')
    
#    fig = plt.figure(1, figsize=(6.5,4))
#    ax =Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=50)
#    
#    scatter = ax.scatter(Data_3d[:,0], Data_3d[:,1], Data_3d[:,2],c=label.to_numpy())
#    legend1 = ax.legend(*scatter.legend_elements(), 
#                        loc='lower left', title='health')
#    ax.add_artist(legend1)
#    
#    ax.w_xaxis.set_ticklabels([])
#    ax.w_yaxis.set_ticklabels([])
#    ax.w_zaxis.set_ticklabels([])
#    ax.set_xlabel('dim1')
#    ax.set_ylabel('dim2')
#    ax.set_zlabel('dim3')
#    ax.set_title('PCA dimension reduced to 3d')
#
#    fig.savefig('asdf.png')
