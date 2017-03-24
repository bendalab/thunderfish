"""
Analysis of Voronoi diagrams and convex hulls based on scipy.spatial

Voronoi diagrams
----------------
voronoi_distances(): Nearest neighbor distances.
voronoi_areas(): The areas of the Voronoi regions for each input point.

Convex hulls
------------
in_hull(): Test if points are within hull.
flatten_simplices(simplices): Transforms list of simplex indices to list of vertex indices.

Bootstrap Voronoi diagrams
--------------------------
random_hull_points(): Generate random points within a hull.
voronoi_bootstrap(): Bootstrapped distances and areas for random point positions.

Usage
-----
Generate 20 random points in 2-D:
'''
import numpy as np
points = np.random.rand(20, 2)
'''

Calculate the Voronoi diagram:
'''
import scipy.spatial as ss
vor = ss.Voronoi(points)
'''

Compute nearest-neighbor distances and Voronoi areas:
'''
distance = voronoi_distances(vor)
areas = voronoi_areas(vor)
'''

Plot Voronoi input points:
'''
plt.plot(vor.points[:,0], vor.points[:,1], 'oc')
'''

Plot vertices of Voronoi regions:
'''
import matplotlib.pyplot as plt
plt.plot(vor.vertices[:,0], vor.vertices[:,1], 'or')
'''

Plot ridges of finite Voronoi regions:
'''
for i, p in enumerate(vor.ridge_vertices):
    if np.all(np.array(p)>=0):
        plt.plot(vor.vertices[p,0], vor.vertices[p,1], 'g', lw=2)
'''

Fill finite Voronoi regions with a color:
'''
for region in vor.regions:
    if not -1 in region:
        polygon = vor.vertices[region]
        plt.fill(*zip(*polygon), alpha=0.5)
plot.show()
'''
(see also http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
 and https://gist.github.com/pv/8036995)


Compute convex hull from Delaunay tesselation:
'''
delaunay = ss.Delaunay(points)
'''

Fill and draw border of convex hull:
'''
vertices = flatten_simplices(delaunay.convex_hull)
plt.fill(delaunay.points[vertices, 0], delaunay.points[vertices, 1], color='black', alpha=0.1)
plt.plot(delaunay.points[vertices, 0], delaunay.points[vertices, 1], 'r', lw=2)
'''

Bootstrap Voronoi distances and areas inside the hull:
'''
d, a = voronoi_bootstrap(vor, hull=delaunay, area_mode='finite')
plt.subplot(2, 1, 1)
plt.hist([ddd for dd in d for ddd in dd], 50)
plt.xlabel('distance')
plt.subplot(2, 1, 2)
plt.hist(a.ravel(), bins=np.arange(0, 0.2, 0.005))
plt.xlabel('area')
'''
... well, actually, instead of making histograms of the flattened data you want to compute percentiles for each bin.
"""

import numpy as np
import scipy.spatial as ss


def voronoi_distances(vor):
    """
    Nearest neighbor distances.
    
    Parameters
    ----------
    vor: scipy.spatial.Voronoi
        The Voronoi class with the data.
    
    Returns
    -------
    distances: array of floats
        For each ridge in vor.ridge_points the distance of the two points
        that are separated by the ridge.
    """
    p1 = vor.points[vor.ridge_points[:,0]]
    p2 = vor.points[vor.ridge_points[:,1]]
    # euclidian distance (Pythagoras):
    distances = np.sqrt(np.sum((p2-p1)**2.0, axis=1))
    return distances


def voronoi_ridge_length(vor):
    """
    Length of Voronoi ridges between nearest neighbors.
    
    Parameters
    ----------
    vor: scipy.spatial.Voronoi
        The Voronoi class with the data.
    
    Returns
    -------
    distances: array of floats
        The length of each ridge in vor.ridge_vertices.
        np.inf if vertex is unknown.
    """
    ridges = np.zeros(len(vor.ridge_vertices))
    for k, p in enumerate(vor.ridge_vertices):
        if np.all(np.array(p)>=0):
            p1 = vor.vertices[p[0]]
            p2 = vor.vertices[p[1]]
            # euclidian distance (Pythagoras) between vertices:
            ridges[k] = np.sqrt(np.sum((p2-p1)**2.0))
        else:
            ridges[k] = np.inf
    return ridges


def voronoi_ridge_areas(vor):
    """
    For each ridge the triangular area of the Voronoi region
    spanned by the center point and the ridge.

    Note:
    -----
    Only two-dimensional data are processed, i.e. vor.ndim must be 2.
        
    Parameters
    ----------
    vor: scipy.spatial.Voronoi
        The Voronoi class with the data.
    
    Returns
    -------
    areas: array of floats
        For each ridge its corresponding triangular area.
        np.inf for infinite ridges.
    """
    ridges = voronoi_ridge_length(vor)
    heights = 0.5*voronoi_distances(vor)
    # area of a triangle:
    areas = 0.5*ridges*heights
    return areas


def voronoi_areas(vor, mode='finite'):
    """
    The areas of the Voronoi regions for each input point.

    Note:
    -----
    Only two-dimensional data are processed, i.e. vor.ndim must be 2.
        
    Parameters
    ----------
    vor: scipy.spatial.Voronoi
        The Voronoi class with the data.
    mode: string
        'full': Calculate area of finite Voronoi regions only,
                set all other to np.nan.
        'finite': Calculate area of all Voronoi regions. From infinite regions
                only areas contributed from finite ridges are considered.
        'all': Calculate area of all Voronoi regions. NOT IMPLEMENTED YET.
    
    Returns
    -------
    areas: array of floats
        For each point its corresponding area.
    """
    ridge_areas = voronoi_ridge_areas(vor)
    areas = np.zeros(len(vor.points))
    if mode == 'full':
        for i in range(len(vor.points)):
            a = 0.0
            for j, rp in enumerate(vor.ridge_points):
                if i in rp:
                    if ridge_areas[j] != np.inf:
                        a += ridge_areas[j]
                    else:
                        a = np.nan
                        break
            areas[i] = a
    elif mode == 'all':
        print('all mode not supported yet!')
        # see http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    else:  # mode == 'finite'
        for i in range(len(vor.points)):
            a = 0.0
            for j, rp in enumerate(vor.ridge_points):
                if i in rp and ridge_areas[j] != np.inf:
                    a += ridge_areas[j]
            areas[i] = a
    return areas


def in_hull(hull, p):
    """
    Test if points p are within hull.

    From http://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl/16898636#16898636

    Parameters
    ----------
    hull: scipy.spatial.Delaunay or scipy.spatial.ConvexHull
        The hull.
    p: 2-D array
        Array of points to be tested.

    Returns
    -------
    inside: array of booleans
        For each point in p whether it is inside the hull.
    """
    if not isinstance(hull, ss.Delaunay):
        hull = ss.Delaunay(hull.points)
    inside = hull.find_simplex(p) >= 0
    return inside


def flatten_simplices(simplices):
    """
    Transforms list of simplex indices to list of vertex indices.

    For example, transforms the Delaunay.convex_hull to a list of points
    of the hull, that can then be easily plotted.
    
    Parameters
    ----------
    simplices: 2-D array of ints
        List of pairs of indices of points forming each ridge of a polygon.

    Returns
    -------
    indices: list of ints
        Indices of vertices of the polygon.
    """
    if len(simplices) == 0:
        return []
    indices = list(simplices[0])
    simplices = np.delete(simplices, 0, 0)
    while len(simplices) > 0:
        for i, s in enumerate(simplices):
            if indices[-1] in s:
                if s[0] == indices[-1]:
                    indices.append(s[1])
                else:
                    indices.append(s[0])
                simplices = np.delete(simplices, i, 0)
                break
    return indices


def random_hull_points(delaunay, n):
    """
    Generate random points within a hull.

    Parameters
    ----------
    delaunay: scipy.spatial.Delaunay
        Delaunay tesselation of some input points that defines the hull.
    n: int
        Number of random points to be generated.

    Returns
    -------
    points: 2-D array
        List of randomly generated points.
    """
    # get bounding box:
    vertices = flatten_simplices(delaunay.convex_hull)
    hull_points = delaunay.points[vertices]
    min_bound = np.min(hull_points, axis=0)
    max_bound = np.max(hull_points, axis=0)
    # generate random points:
    points = np.zeros((0, delaunay.ndim))
    while len(points) < n:
        # random points within bounding box:
        newpoints = np.random.rand(n/2, delaunay.ndim)
        newpoints *= max_bound - min_bound
        newpoints += min_bound
        # only take the ones within hull:
        inside = in_hull(delaunay, newpoints)
        points = np.vstack((points, newpoints[inside]))
    return points[:n]


def voronoi_bootstrap(vor, n=1000, hull=None, area_mode='finite'):
    """
    Bootstrapped distances and areas for random point positions.

    Same number of points as in vor, randomly placed in the same region
    as given by vor.min_bound and vor.max_bound.
        
    Parameters
    ----------
    vor: scipy.spatial.Voronoi
        The Voronoi class with the original data.
    n: int
        The number of bootstraps.
    hull: scipy.spatial.Delaunay or None
        If not None then place random points within hull of the points
        of hull.
    area_mode: string
        Mode string passed to voronoi_areas().
    
    Returns
    -------
    distances: list of 1-D array of floats
        The bootstrapped distances of nearest neighbors.
    areas: n x vor.npoints array of floats
        The bootstrapped Voronoi areas.
    """
    distances = []
    areas = []
    for k in range(n):
        # random points:
        if hull is not None:
            points = random_hull_points(hull, vor.npoints)
        else:
            points = np.random.rand(vor.npoints, vor.ndim)
            d = vor.max_bound - vor.min_bound
            points *= d
            points += vor.min_bound
        # Voronoi:
        vvor = ss.Voronoi(points)
        distances.append(voronoi_distances(vvor))
        areas.append(voronoi_areas(vvor, area_mode))
    return distances, np.array(areas)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    print("Checking spatial module ...")

    # generate random points:
    n = 20
    points = np.random.rand(n, 2)

    # compute convex hull:
    hull = ss.ConvexHull(points)
    # requires scipy 0.17:
    if hasattr(hull, 'area'):
        print('hull area: %g' % hull.area)
        print('hull volume: %g' % hull.volume)

    plt.title('ConvexHull')
    # plot points:
    plt.scatter(*points.T, s=100)
    # fill hull:
    plt.fill(hull.points[hull.vertices, 0], hull.points[hull.vertices, 1], color='black', alpha=0.1)
    # plot hull border:
    plt.fill(hull.points[hull.vertices, 0], hull.points[hull.vertices, 1], color='none', edgecolor='r', lw=2)

    # compute hull from Delaunay tesselation:
    delaunay = ss.Delaunay(points)

    plt.figure()
    plt.title('Delaunay with random points inside hull')
    # plot points:
    plt.scatter(*points.T, s=100)
    # plot triangulation:
    poly = PolyCollection(delaunay.points[delaunay.vertices], facecolors='none', edgecolors='c')
    plt.gca().add_collection(poly)
    # fill hull:
    poly = PolyCollection(delaunay.points[delaunay.vertices], facecolors='black', edgecolor='black', alpha=0.1)
    plt.gca().add_collection(poly)
    # plot hull border:
    for simplex in delaunay.convex_hull:
        plt.plot(delaunay.points[simplex, 0], delaunay.points[simplex, 1], 'r-', lw=2)

    # random points inside hull:
    other = random_hull_points(delaunay, 20)
    plt.scatter(*other.T, c='g', s=60)

    plt.figure()
    plt.title('Delaunay flattened_simplices with random points')
    # plot points:
    plt.scatter(*points.T, s=100)
    vertices = flatten_simplices(delaunay.convex_hull)
    # fill hull:
    plt.fill(delaunay.points[vertices, 0], delaunay.points[vertices, 1], color='black', alpha=0.1)
    # plot hull border:
    plt.plot(delaunay.points[vertices, 0], delaunay.points[vertices, 1], 'r', lw=2)

    # generate more points:
    other = np.random.rand(n, 2)*1.5 - 0.25
    # check which of them are inside the hull:
    inside = in_hull(delaunay, other)
    # plot inside points:
    plt.scatter(other[inside, 0], other[inside, 1], c='r', s=60)
    # plot outside points:
    plt.scatter(other[~inside, 0], other[~inside, 1], c='g', s=60)

    
    # calculate Voronoi diagram:
    vor = ss.Voronoi(points)
    
    # what we get is:
    print('dimension: %d' % vor.ndim)
    print('number of points: %d' % vor.npoints)
    print('distances of nearest neighbors:')
    print(voronoi_distances(vor))
    print('length of corresponding ridges:')
    print(voronoi_ridge_length(vor))
    print('area of corresponding triangles:')
    print(voronoi_ridge_areas(vor))
    print('Voronoi area of each point:')
    print(voronoi_areas(vor))

    # plot voronoi diagram:
    ss.voronoi_plot_2d(vor)
    plt.title('voronoi_plot_2d')
    plt.xlim(vor.min_bound[0]-0.5, vor.max_bound[0]+0.5)
    plt.ylim(vor.min_bound[1]-0.5, vor.max_bound[1]+0.5)

    # plot what we understand:
    plt.figure()
    plt.title('Voronoi')
    # voronoi points:
    plt.plot(vor.points[:,0], vor.points[:,1], 'oc')
    for i, p in enumerate(vor.points):
        plt.text(p[0], p[1]+0.05, 'p%d' % i)
    # voronoi vertices:
    plt.plot(vor.vertices[:,0], vor.vertices[:,1], 'or')
    for i, p in enumerate(vor.vertices):
        plt.text(p[0], p[1]+0.05, 'v%d' % i)
    # voronoi regions:
    ## for i, p in enumerate(vor.regions):
    ##     if len(p) > 0 and np.all(np.array(p)>=0):
    ##         print('region %d %s' % (i, str(p)))
    ##         plt.plot(vor.vertices[p,0], vor.vertices[p,1], 'g')
    ## for i, p in enumerate(vor.points):
    ##     r = vor.point_region[i]
    ##     pr = vor.regions[r]
    ##     if len(pr) > 0 and np.all(np.array(pr)>=0):
    ##         plt.text(p[0], p[1]+0.2, 'region %d for point %d' % (r, i), ha='center')
    # voronoi ridges:
    for i, p in enumerate(vor.ridge_vertices):
        if np.all(np.array(p)>=0):
            plt.plot(vor.vertices[p,0], vor.vertices[p,1], 'g', lw=2)
    # colorize:
    for region in vor.regions:
        if not -1 in region:
            polygon = vor.vertices[region]
            plt.fill(*zip(*polygon), alpha=0.5)
    # see also http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    # and https://gist.github.com/pv/8036995
    plt.xlim(vor.min_bound[0]-0.5, vor.max_bound[0]+0.5)
    plt.ylim(vor.min_bound[1]-0.5, vor.max_bound[1]+0.5)

    # bootstrap:
    print('bootstrap bounding box ...')
    db, ab = voronoi_bootstrap(vor, hull=None, area_mode='finite')
    print('bootstrap hull ...')
    dd, ad = voronoi_bootstrap(vor, hull=delaunay, area_mode='finite')
    # also try
    #db, ab = voronoi_bootstrap(vor, hull=None, area_mode='full')
    #dd, ad = voronoi_bootstrap(vor, hull=delaunay, area_mode='full')
    print('... done.')

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.hist([d for dd in db for d in dd], 50)
    plt.title('bootstrap bounding box')
    plt.xlabel('distance')
    plt.subplot(2, 2, 2)
    plt.hist(ab.ravel(), bins=np.arange(0, 0.2, 0.005))
    plt.title('bootstrap bounding box')
    plt.xlabel('area')
    plt.subplot(2, 2, 3)
    plt.hist([d for dd in db for d in dd], 50)
    plt.title('bootstrap hull')
    plt.xlabel('distance')
    plt.subplot(2, 2, 4)
    plt.hist(ad.ravel(), bins=np.arange(0, 0.2, 0.005))
    plt.title('bootstrap hull')
    plt.xlabel('area')
    plt.tight_layout()

    plt.show()

