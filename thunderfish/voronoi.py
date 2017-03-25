import numpy as np
import scipy.spatial as ss

class Voronoi:
    
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
        self.vor = ss.Voronoi(points, furthest_site=False, incremental=False,
                              qhull_options=qhull_options)
        self.hull = ss.Delaunay(points, furthest_site=False, incremental=False,
                                qhull_options=qhull_options)
        self.outer_hull = ss.Delaunay(np.vstack((self.vor.points, self.vor.vertices)),
                                      furthest_site=False, incremental=False,
                                      qhull_options=qhull_options)
        self.ndim = self.vor.ndim
        self.npoints = self.vor.npoints
        self.points = self.vor.points
        self.vertices = self.vor.vertices
        self.regions = self.vor.regions
        self.ridge_points = self.vor.ridge_points
        self.ridge_vertices = self.vor.ridge_vertices
        self.min_bound = self.vor.min_bound
        self.max_bound = self.vor.max_bound
        self.hull_vertices = self._flatten_simplices(self.hull.convex_hull)
        self.outer_hull_vertices = self._flatten_simplices(self.outer_hull.convex_hull)


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


    def distances(self):
        """
        Nearest neighbor distances.

        Returns
        -------
        distances: array of floats
            For each ridge in vor.ridge_points the distance of the two points
            that are separated by the ridge.
        """
        p1 = self.points[self.ridge_points[:,0]]
        p2 = self.points[self.ridge_points[:,1]]
        return ss.minkowski_distance(p1, p2)


    def ridge_lengths(self):
        """
        Length of Voronoi ridges between nearest neighbors.

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
                ridges[k] = ss.minkowski_distance(p1, p2)
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
        heights = 0.5*self.distances()
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
            'finite': Calculate area of all Voronoi regions. From infinite regions
                    only areas contributed from finite ridges are considered.
            'all': Calculate area of all Voronoi regions. NOT IMPLEMENTED YET.

        Returns
        -------
        areas: array of floats
            For each point its corresponding area.
        """
        ridge_areas = self.ridge_areas()
        areas = np.zeros(len(self.vor.points))
        if mode == 'full':
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
        elif mode == 'all':
            print('all mode not supported yet!')
            # see http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
        else:  # mode == 'finite'
            for i in range(len(self.vor.points)):
                a = 0.0
                for j, rp in enumerate(self.vor.ridge_points):
                    if i in rp and ridge_areas[j] != np.inf:
                        a += ridge_areas[j]
                areas[i] = a
        return areas
    

    def nfinite_regions(self):
        """
        Number of finite Voronoi regions.

        Returns
        -------
        nregions: int
            The number of finite Voronoi regions.
        """
        nregions = 0
        for region in self.vor.regions:
            if len(region) > 0 and not -1 in region:
                nregions += 1
        return nregions
    

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
                ax.plot(self.vertices[p,0], self.vertices[p,1], **kwargs)

    def fill_regions(self, ax=None, colors=None, **kwargs):
        """
        Fill each finite region of the Voronoi diagram with a color.

        See also http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
        and https://gist.github.com/pv/8036995

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
    n = 10
    points = np.random.rand(n, 2)
    
    # calculate Voronoi diagram:
    vor = Voronoi(points)
    
    # what we get is:
    print('dimension: %d' % vor.ndim)
    print('number of points: %d' % vor.npoints)
    print('number of finite Voronoi regions: %d' % vor.nfinite_regions())
    print('points enclosing ridges:')
    print(vor.ridge_points)
    print('distances of nearest neighbors:')
    print(vor.distances())
    print('length of corresponding ridges:')
    print(vor.ridge_lengths())
    print('area of corresponding triangles:')
    print(vor.ridge_areas())
    print('Voronoi area of each point:')
    print(vor.areas())

    # plot Voronoi diagram:
    plt.title('Voronoi')
    vor.fill_outer_hull(color='black', alpha=0.2)
    vor.plot_outer_hull(color='m', lw=2)
    vor.fill_hull(color='black', alpha=0.1)
    vor.plot_hull(color='r', lw=2)
    vor.fill_regions(colors=['red', 'green', 'blue', 'orange', 'cyan'], alpha=0.3)
    vor.plot_points(text='p%d', c='c', s=100)
    vor.plot_ridges(c='g', lw=2)
    vor.plot_vertices(text='v%d', c='r', s=60)
    #plt.xlim(vor.min_bound[0]-0.5, vor.max_bound[0]+0.5)
    #plt.ylim(vor.min_bound[1]-0.5, vor.max_bound[1]+0.5)
    plt.axes().set_aspect('equal')

    plt.show()
    exit()

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

    plt.show()
