"""
Analysis of Voronoi diagrams based on scipy.spatial.

class Voronoi: Compute and analyse Voronoi diagrams.
"""


import numpy as np
import scipy.spatial as sp


class Voronoi:
    """
    Voronoi diagram
    ---------------
    distances(): Nearest neighbor distances.
    ridge_lengths(): Length of Voronoi ridges between nearest neighbors.
    areas(): The areas of the Voronoi regions for each input point.
    point_types(): The type of Voronoi area (infinite, finite, inside)
    
    Convex hull
    -----------
    in_hull(): Test if points are within hull.
    
    Bootstrap Voronoi diagrams
    --------------------------
    random_points(): Generate random points.
    bootstrap(): Bootstrapped distances and areas for random point positions.

    Plotting
    --------
    plot_points(): Plot and optionally annotate the input points of the Voronoi diagram.
    plot_vertices(): Plot and optionally annotate the vertices of the Voronoi diagram.
    plot_distances(): Plot lines connecting the nearest neighbors in the Voronoi diagram.
    plot_ridges(): Plot the finite ridges of the Voronoi diagram.
    fill_regions(): Fill each finite region of the Voronoi diagram with a color.
    plot_hull(): Plot the hull line containing the input points.
    fill_hull(): Fill the hull containing the input points with a color.
    
    Usage
    -----
    Generate 20 random points in 2-D:
    '''
    import numpy as np
    points = np.random.rand(20, 2)
    '''
    
    Calculate the Voronoi diagram:
    '''
    from thunderfish.voronoi import Voronoi
    vor = Voronoi(points)
    '''
    
    Compute nearest-neighbor distances and Voronoi areas:
    '''
    distance = vor.distances()
    areas = vor.areas()
    '''
    
    Plot Voronoi areas, distances and input points:
    '''
    import matplotlib.pyplot as plt
    vor.fill_regions(colors=['red', 'green', 'blue', 'orange', 'cyan'], alpha=0.3)
    vor.plot_distances(color='red')
    vor.plot_points(text='p%d', c='c', s=100)
    '''
    """
    
    def __init__(self, points, qhull_options=None):
        """
        Compute the Voronoi diagram and the convex hull for a set of points.

        Parameter
        ---------
        points: list of lists of floats
            List of point coordiantes.
        qhull_options: string or None
            Options to be passed on to QHull. From the manual:
            Qbb  - scale last coordinate to [0,m] for Delaunay triangulations
            Qc   - keep coplanar points with nearest facet      
            Qx   - exact pre-merges (skips coplanar and anglomaniacs facets)
            Qz   - add point-at-infinity to Delaunay triangulation
            QJn  - randomly joggle input in range [-n,n]
            Qs   - search all points for the initial simplex
            Qz   - add point-at-infinity to Voronoi diagram
            QGn  - Voronoi vertices if visible from point n, -n if not
            QVn  - Voronoi vertices for input point n, -n if not
            Default is: "Qbb Qc Qz Qx" for ndim > 4 and "Qbb Qc Qz" otherwise.
        """
        self.vor = sp.Voronoi(points, furthest_site=False, incremental=False,
                              qhull_options=qhull_options)
        self.hull = sp.Delaunay(points, furthest_site=False, incremental=False,
                                qhull_options=qhull_options)
        self.outer_hull = sp.Delaunay(np.vstack((self.vor.points, self.vor.vertices)),
                                      furthest_site=False, incremental=False,
                                      qhull_options=qhull_options)
        self._compute_distances()
        self._compute_infinite_vertices()
        self.ndim = self.vor.ndim
        self.npoints = self.vor.npoints
        self.points = self.vor.points
        self.vertices = self.vor.vertices
        self.regions = self.vor.regions
        self.ridge_points = self.vor.ridge_points
        self.ridge_vertices = self.vor.ridge_vertices
        self.min_bound = self.vor.min_bound
        self.max_bound = self.vor.max_bound
        self.inside_vertices = self.in_hull(self.vertices)
        self.hull_vertices = self._flatten_simplices(self.hull.convex_hull)
        self.outer_hull_vertices = self._flatten_simplices(self.outer_hull.convex_hull)


    def _compute_distances(self):
        """
        Compute distances between points.
        """
        # For each ridge the distance of the points enclosing the ridge:
        p1 = self.vor.points[self.vor.ridge_points[:,0]]
        p2 = self.vor.points[self.vor.ridge_points[:,1]]
        self.ridge_distances =  sp.minkowski_distance(p1, p2)

        # For each point all its Voronoi distances:
        self.neighbor_points = [[] for k in range(len(self.vor.points))]
        self.neighbor_distances = [[] for k in range(len(self.vor.points))]
        for dist, points in zip(self.ridge_distances, self.vor.ridge_points):
            self.neighbor_points[points[0]].append(points[1])
            self.neighbor_points[points[1]].append(points[0])
            self.neighbor_distances[points[0]].append(dist)
            self.neighbor_distances[points[1]].append(dist)
        for k in range(len(self.neighbor_points)):
            inx = np.argsort(self.neighbor_distances[k])
            self.neighbor_points[k] = np.array(self.neighbor_points[k])[inx]
            self.neighbor_distances[k] = np.array(self.neighbor_distances[k])[inx]

        # For each point the distance to its neares neighbor:
        self.nearest_distances = np.zeros(len(self.neighbor_distances))
        for k in range(len(self.neighbor_distances)):
            self.nearest_distances[k] = self.neighbor_distances[k][0]

    def _compute_infinite_vertices(self):
        center = self.vor.points.mean(axis=0)
        ptp_bound = self.vor.points.ptp(axis=0)
        self.infinite_vertices = []
        for points, vertices in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            vertices = np.asarray(vertices)
            if np.all(vertices >= 0):
                self.infinite_vertices.append([])
            else:
                i = vertices[vertices >= 0][0]  # finite end Voronoi vertex
                t = self.vor.points[points[1]] - self.vor.points[points[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = self.vor.points[points].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.vor.vertices[i] + direction * ptp_bound.max()
                self.infinite_vertices.append(far_point)

    def _flatten_simplices(self, simplices):
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


    def in_hull(self, p):
        """
        Test if points p are within hull.

        From http://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl/16898636#16898636

        Parameters
        ----------
        p: 2-D array
            Array of points to be tested.

        Returns
        -------
        inside: array of booleans
            For each point in p whether it is inside the hull.
        """
        inside = self.hull.find_simplex(p) >= 0
        return inside


    def remove_vertices_outside_hull(self):
        inside = self.in_hull(self.vertices)
        rm_inx = np.arange(len(vor.vertices))[~inside]
        vert_map = {-1: -1}
        c = 0
        for k in range(len(self.ridge_vertices)):
            if k in rm_inx:
                vert_map[k] = -1
            else:
                vert_map[k] = c
                c += 1
        self.vertices = np.delete(self.vertices, rm_inx, 0)
        for ridge in self.ridge_vertices:
            for k in range(len(ridge)):
                ridge[k] = vert_map[ridge[k]]
        for region in self.regions:
            for k in range(len(region)):
                region[k] = vert_map[region[k]]
        self.outer_hull = self.hull
        self.outer_hull_vertices = self.hull_vertices


    def point_types(self):
        """
        The type of the Voronoi regions for each input point.

        Returns
        -------
        points: array of ints
            For each point 1: finite region with all vertices inside hull, 0: finite regions,
                          -1: infinite regions
        """
        points = np.ones(len(self.vor.points), dtype=int)
        for i in range(len(self.vor.points)):
            for j, rp in enumerate(self.vor.ridge_points):
                if i in rp:
                    if not np.all(self.inside_vertices[self.vor.ridge_vertices[j]]) and\
                      points[i] > 0:
                        points[i] = 0
                    if not np.all(np.array(self.vor.ridge_vertices[j]) >= 0) and\
                      points[i] > -1:
                        points[i] = -1
        return points


    def ridge_lengths(self):
        """
        Length of Voronoi ridges between nearest neighbors.

        May be used, for example, as a weigth for distances().
        XXX How to deal with ridges with vertices outside the hull?

        Returns
        -------
        distances: array of floats
            The length of each ridge in vor.ridge_vertices.
            np.inf if vertex is unknown.
        """
        ridges = np.zeros(len(self.vor.ridge_vertices))
        for k, p in enumerate(self.vor.ridge_vertices):
            if np.all(np.array(p)>=0):
                p1 = self.vor.vertices[p[0]]
                p2 = self.vor.vertices[p[1]]
                ridges[k] = sp.minkowski_distance(p1, p2)
            else:
                ridges[k] = np.inf
        return ridges


    def ridge_areas(self):
        """
        For each ridge the triangular area of the Voronoi region
        spanned by the center point and the ridge.

        Note:
        -----
        Only two-dimensional data are processed, i.e. vor.ndim must be 2.

        Returns
        -------
        areas: array of floats
            For each ridge its corresponding triangular area.
            np.inf for infinite ridges.
        """
        ridges = self.ridge_lengths()
        heights = 0.5*self.ridge_distances
        # area of a triangle:
        areas = 0.5*ridges*heights
        return areas

    
    def areas(self, mode='finite'):
        """
        The areas of the Voronoi regions for each input point.

        Note:
        -----
        Only two-dimensional data are processed, i.e. vor.ndim must be 2.

        Parameters
        ----------
        mode: string
            'full': Calculate area of finite Voronoi regions only,
                    set all other to np.nan.
            'inside': Calculate area of finite Voronoi regions
                      whose vertices are all inside the hull,
                      set all other to np.nan.
            'finite': Calculate area of all Voronoi regions. From infinite regions
                    only areas contributed by finite ridges are considered.
            'finite_inside': Calculate area of all Voronoi regions.
                    Consider only areas of finite ridges
                    whose vertices are all inside the hull.

        Returns
        -------
        areas: array of floats
            For each point its corresponding area.
        """
        ridge_areas = self.ridge_areas()
        areas = np.zeros(len(self.vor.points))
        if mode == 'inside':
            for i in range(len(self.vor.points)):
                a = 0.0
                for j, rp in enumerate(self.vor.ridge_points):
                    if i in rp:
                        if ridge_areas[j] != np.inf and \
                          np.all(self.inside_vertices[self.vor.ridge_vertices[j]]):
                            a += ridge_areas[j]
                        else:
                            a = np.nan
                            break
                areas[i] = a
        elif mode == 'full':
            for i in range(len(self.vor.points)):
                a = 0.0
                for j, rp in enumerate(self.vor.ridge_points):
                    if i in rp:
                        if ridge_areas[j] != np.inf:
                            a += ridge_areas[j]
                        else:
                            a = np.nan
                            break
                areas[i] = a
        elif mode == 'finite_inside':
            for i in range(len(self.vor.points)):
                a = 0.0
                for j, rp in enumerate(self.vor.ridge_points):
                    if i in rp and ridge_areas[j] != np.inf and \
                      np.all(self.inside_vertices[self.vor.ridge_vertices[j]]):
                        a += ridge_areas[j]
                areas[i] = a
        elif mode == 'finite':
            for i in range(len(self.vor.points)):
                a = 0.0
                for j, rp in enumerate(self.vor.ridge_points):
                    if i in rp and ridge_areas[j] != np.inf:
                        a += ridge_areas[j]
                areas[i] = a
        else:
            print('')            
            print('Voronoi.areas(): unknown value "%s" for the mode parameter:' % mode)
            print('Use one of the following values:')
            print('  full: Finite Voronoi regions only.')
            print('  inside: Finite Voronoi regions whose vertices are all inside the hull.')
            print('  finite: Use all areas corresponding to finite ridges.')
            print('  finite_inside: Use all areas corresponding to finite ridges whose vertices are all inside the hull.')
            print('')            
        return areas


    def hull_area(self):
        """
        The area of the convex hull of the input points.
        
        Returns
        -------
        area: float
            The area of the convex hull.
        """
        # two sides of the simplex triangles:
        ab = self.hull.points[self.hull.simplices[:,0],:] - self.hull.points[self.hull.simplices[:,1],:]
        cb = self.hull.points[self.hull.simplices[:,2],:] - self.hull.points[self.hull.simplices[:,1],:]
        # area of each simplex is half of the absolute value of the cross product:
        area = 0.5*np.sum(np.abs(np.cross(ab, cb)))
        return area
    

    def random_points(self, n=None, mode='bbox'):
        """
        Generate random points.

        Parameters
        ----------
        n: int or None
            Number of random points to be generated.
            If None n is set to the number of points in the Voronoi diagram.
        mode: string
            'bbox' place points randomly in rectangular bounding box of the Voronoi diagram.
            'hull' place points randomly within hull of input data.

        Returns
        -------
        points: 2-D array
            List of randomly generated points.
        """
        # number of points:
        if n is None:
            n = self.npoints
        # get bounding box:
        min_bound = np.min(self.points, axis=0)
        max_bound = np.max(self.points, axis=0)
        # generate random points:
        points = np.zeros((0, self.ndim))
        while len(points) < n:
            # random points within bounding box:
            newpoints = np.random.rand(n/2, self.ndim)
            newpoints *= max_bound - min_bound
            newpoints += min_bound
            if mode == 'hull':
                # only take the ones within hull:
                inside = vor.in_hull(newpoints)
                points = np.vstack((points, newpoints[inside]))
            elif mode == 'bbox':
                points = np.vstack((points, newpoints))
            else:
                print('')            
                print('Voronoi.random_points(): unknown value "%s" for the mode parameter:' % mode)
                print('Use one of the following values:')
                print('  bbox: Place points within rectangular bounding box.')
                print('  hull: Place points inside the hull.')
                print('')
                return
        return points[:n]


    def bootstrap(self, n=1000, mode='bbox', area_mode='finite'):
        """
        Bootstrapped distances and areas for random point positions.

        Parameters
        ----------
        n: int
            The number of bootstraps.
        mode: string
            Mode string passed to random_points().
        area_mode: string
            Mode string passed to areas().

        Returns
        -------
        distances: list of 1-D array of floats
            The bootstrapped distances of nearest neighbors.
        areas: n x npoints array of floats
            The bootstrapped Voronoi areas.
        """
        distances = []
        areas = []
        for k in range(n):
            # random points:
            points = self.random_points(mode=mode)
            # Voronoi:
            vor = Voronoi(points)
            distances.append(vor.ridge_distances)
            areas.append(vor.areas(area_mode))
        return distances, np.array(areas)


    def plot_points(self, ax=None, text=None, text_offs=(0, 0.05), text_align='center',
                    **kwargs):
        """
        Plot and optionally annotate the input points of the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        text: string or None
            If not None the string that is placed at each point.
            A '%d' is replaced by the index of the point.
        text_offset: tuple of numbers
            The offset of the point labels.
        text_align: string
            The horizontal alignment of the point labels.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.scatter() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.points[:,0], self.points[:,1], **kwargs)
        if text is not None:
            for i, p in enumerate(self.points):
                s = text
                if '%' in text:
                    s = text % i
                ax.text(p[0]+text_offs[0], p[1]+text_offs[1], s, ha=text_align)
        
    def plot_vertices(self, ax=None, text=None, text_offs=(0, 0.05), text_align='center',
                      **kwargs):
        """
        Plot and optionally annotate the vertices of the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        text: string or None
            If not None the string that is placed at each vertex.
            A '%d' is replaced by the index of the vertex.
        text_offset: tuple of numbers
            The offset of the vertex labels.
        text_align: string
            The horizontal alignment of the vertex labels.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.scatter() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.vertices[:,0], self.vertices[:,1], **kwargs)
        if text is not None:
            for i, p in enumerate(self.vertices):
                s = text
                if '%' in text:
                    s = text % i
                ax.text(p[0]+text_offs[0], p[1]+text_offs[1], s, ha=text_align)

    def plot_distances(self, ax=None, **kwargs):
        """
        Plot lines connecting the nearest neighbors in the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        for i, p in enumerate(self.ridge_points):
            ax.plot(self.points[p, 0], self.points[p, 1], **kwargs)

    def plot_ridges(self, ax=None, **kwargs):
        """
        Plot the finite ridges of the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        for i, p in enumerate(self.ridge_vertices):
            if np.all(np.array(p)>=0):
                ax.plot(self.vertices[p, 0], self.vertices[p, 1], **kwargs)

    def plot_infinite_ridges(self, ax=None, **kwargs):
        """
        Plot the infinite ridges of the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        for far_point, vertices in zip(self.infinite_vertices, self.vor.ridge_vertices):
            vertices = np.asarray(vertices)
            if not np.all(vertices >= 0):
                i = vertices[vertices >= 0][0]  # finite end Voronoi vertex
                ax.plot([self.vor.vertices[i][0], far_point[0]],
                        [self.vor.vertices[i][1], far_point[1]], **kwargs)

    def fill_regions(self, ax=None, colors=None, **kwargs):
        """
        Fill each finite region of the Voronoi diagram with a color.
        
        See also http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        colors: list of colors or None
            If not None then these colors are used in turn to fill the regions.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.fill() function.
        """
        if ax is None:
            ax = plt.gca()
        c = 0
        print self.regions
        for region in self.regions:
            if not -1 in region:
                polygon = self.vertices[region]
                if len(polygon) > 0:
                    c += 1
                    if colors is None:
                        ax.fill(polygon[:, 0], polygon[:, 1], **kwargs)
                    else:
                        ax.fill(polygon[:, 0], polygon[:, 1],
                                color=colors[c % len(colors)], **kwargs)
        
    def plot_hull(self, ax=None, **kwargs):
        """
        Plot the hull line containing the input points.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.hull.points[self.hull_vertices, 0],
                self.hull.points[self.hull_vertices, 1], **kwargs)

    def fill_hull(self, ax=None, **kwargs):
        """
        Fill the hull containing the input points with a color.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.fill() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.fill(self.hull.points[self.hull_vertices, 0],
                self.hull.points[self.hull_vertices, 1], **kwargs)
        
    def plot_outer_hull(self, ax=None, **kwargs):
        """
        Plot the hull line containing the input points and the vertices of the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.outer_hull.points[self.outer_hull_vertices, 0],
                self.outer_hull.points[self.outer_hull_vertices, 1], **kwargs)

    def fill_outer_hull(self, ax=None, **kwargs):
        """
        Fill the hull containing the input points and the vertices of the Voronoi diagram.

        Parameter
        ---------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.fill() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.fill(self.outer_hull.points[self.outer_hull_vertices, 0],
                self.outer_hull.points[self.outer_hull_vertices, 1], **kwargs)

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    print("Checking voronoi module ...")

    # generate random points:
    rs = np.random.randint(0xffffffff)
    #rs = 3550922155
    print('random seed: %ld' % rs)
    np.random.seed(rs)
    n = 10
    points = np.random.rand(n, 2)
    
    # calculate Voronoi diagram:
    vor = Voronoi(points)
    
    # what we get is:
    print('dimension: %d' % vor.ndim)
    print('number of points: %d' % vor.npoints)
    print('area of convex hull: %g' % vor.hull_area())
    print('')
    print('distances of nearest neighbors:')
    print(vor.nearest_distances)
    print('for each point all Voronoi distances:')
    print(vor.neighbor_distances)
    print('for each point all its neighbors:')
    print(vor.neighbor_points)
    print('for each ridge distances of neighbors:')
    print(vor.ridge_distances)
    print('corresponding neighbors enclosing ridges:')
    print(vor.ridge_points)
    print('')
    print('length of corresponding ridges:')
    print(vor.ridge_lengths())
    print('area of corresponding triangles:')
    print(vor.ridge_areas())
    print('Voronoi area of each point (full):')
    print(vor.areas('full'))
    print('Voronoi area of each point (inside):')
    print(vor.areas('inside'))
    print('Voronoi area of each point (finite):')
    print(vor.areas('finite'))
    print('Voronoi area of each point (finite_inside):')
    print(vor.areas('finite_inside'))
    print('Type of Voronoi area of each point:')
    print(vor.point_types())

    # plot Voronoi diagram:
    plt.title('Voronoi')
    #vor.fill_outer_hull(color='black', alpha=0.2)
    #vor.plot_outer_hull(color='m', lw=2)
    vor.fill_hull(color='black', alpha=0.1)
    #vor.plot_hull(color='r', lw=2)
    vor.fill_regions(colors=['red', 'green', 'blue', 'orange', 'cyan'], alpha=0.3)
    #vor.plot_distances(color='red')
    vor.plot_points(text='p%d', c='c', s=100)
    vor.plot_ridges(c='g', lw=2)
    vor.plot_infinite_ridges(c='g', lw=2, linestyle='dashed')
    #vor.plot_vertices(text='v%d', c='r', s=60)
    plt.xlim(vor.min_bound[0]-0.2, vor.max_bound[0]+0.2)
    plt.ylim(vor.min_bound[1]-0.2, vor.max_bound[1]+0.2)
    plt.axes().set_aspect('equal')

    """
    vor.remove_vertices_outside_hull()
    plt.figure()
    plt.title('Voronoi inside hull')
    vor.fill_hull(color='black', alpha=0.2)
    vor.plot_hull(color='r', lw=2)
    vor.fill_regions(colors=['red', 'green', 'blue', 'orange', 'cyan'], alpha=0.3)
    vor.plot_points(text='p%d', c='c', s=100)
    vor.plot_ridges(c='g', lw=2)
    vor.plot_vertices(text='v%d', c='r', s=60)
    #plt.xlim(vor.min_bound[0]-0.5, vor.max_bound[0]+0.5)
    #plt.ylim(vor.min_bound[1]-0.5, vor.max_bound[1]+0.5)
    plt.axes().set_aspect('equal')
    """

    #hull = sp.Delaunay(vor.points[vor.hull_vertices[:-1]])

    plt.show()
    exit()

    # bootstrap:
    print('bootstrap bounding box ...')
    db, ab = vor.bootstrap(mode='bbox', area_mode='finite')
    print('bootstrap hull ...')
    dd, ad = vor.bootstrap(mode='hull', area_mode='finite')
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

