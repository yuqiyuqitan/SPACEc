import matplotlib.pyplot as plt


def tissue_lables(tissueframe, region = 'region1'):
    centroids = tissueframe.groupby('tissue').mean()
    fig, ax = plt.subplots()
    ax.scatter(centroids['x'], centroids['y'])
    ax.invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    
    for i, txt in enumerate(centroids.index):
        ax.annotate(txt, (list(centroids['x'])[i], list(centroids['y'])[i]))
    
    plt.title('Tissue piece labels')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(centroids['x'], centroids['y'])
    ax.invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    
    for i, txt in enumerate(centroids[region]):
        ax.annotate(int(txt), (list(centroids['x'])[i], list(centroids['y'])[i]))
    
    plt.title('Region labels')
    plt.show()
