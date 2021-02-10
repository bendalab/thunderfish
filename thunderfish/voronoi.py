"""
Analyze Voronoi diagrams based on scipy.spatial.

## Classes

- `class Voronoi`: Compute and analyse Voronoi diagrams.
"""


import numpy as np
import scipy.spatial as sp
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


class Voronoi(object):
    """
    Parameters
    ----------
    points: ndarray of floats, shape (npoints, 2)
        List of point coordinates.
    radius: float or None
        Radius for computing far points of infinite ridges.
        If None twice the maximum extent of the input points is used.
    qhull_options: string or None
        Options to be passed on to QHull. From the manual:
        
        - Qbb  - scale last coordinate to [0,m] for Delaunay triangulations
        - Qc   - keep coplanar points with nearest facet      
        - Qx   - exact pre-merges (skips coplanar and anglomaniacs facets)
        - Qz   - add point-at-infinity to Delaunay triangulation
        - QJn  - randomly joggle input in range [-n,n]
        - Qs   - search all points for the initial simplex
        - Qz   - add point-at-infinity to Voronoi diagram
        - QGn  - Voronoi vertices if visible from point n, -n if not
        - QVn  - Voronoi vertices for input point n, -n if not
        
        Default is: "Qbb Qc Qz Qx" for ndim > 4 and "Qbb Qc Qz" otherwise.
            
    Input points
    ------------
    The points from which the Voronoi diagram is constructed.
    
    ndim: int
        The dimension of the input points, i.e. number of coordinates.
    npoints: int
        The number of input points.
    points: 2-d ndarray
        List of input point coordinates.
    center: ndarray of floats
        Center of mass of the input points (i.e. mean coordinates).
    min_bound: ndarray of floats
        Lower left corner of the bounding-box of the input points.
    max_bound: ndarray of floats
        Upper right corner of the bounding-box of the input points.
    
    Distances between input points
    ------------------------------
    ridge_points: 2-d ndarray of ints
        List of pairs of indices to `points` enclosing a Voronoi ridge.
    ridge_distances: ndarray of floats
        For each ridge in `ridge_points` the distance between the enclosing points.
    neighbor_points: list of ndarrays of ints
        For each point in `points` a list of indices of the Voronoi-cell's neighboring points.
    neighbor_distances: list of ndarrays of floats
        For each point in `points` a list of distances to the Voronoi-cell's neighboring points,
        matching `neighbor_points`.
    nearest_points: list of ints
        For each point in `points` the index of its nearest neighbor.
    nearest_distances: ndarray of floats
        For each point in `points` the distance to its nearest neighbor.
    mean_nearest_distance: float
        Unbiased estimate of the mean nearest-neighbor distance. This is
        the average nearest-neighbor distance corrected by one s.e.m,
        since points close to the border of the region have too large
        nearest-neighbor distances.
    
    Voronoi diagram
    ---------------
    vertices: 2-d ndarray of floats
        List of vertex coordinates enclosing the Voronoi regions.
    regions: list of list of ints
        List of lists of indices to vertices in `vertices` making up each Voronoi region.
    ridge_vertices: list of list of ints
        List of lists of indices to `vertices` making up a Voronoi ridge.
        The pairs of `points` to poth sides of the ridge are listed in `ridge_points`.
    ridge_lengths(): Length of Voronoi ridges between nearest neighbors.
    areas(): The areas of the Voronoi regions for each input point in `points`.
    point_types(): The type of Voronoi area (infinite, finite, inside) for each input point.
    inside_vertices: list of ints
        Indices of `vertices` that are inside the convex hull of the input points.
    infinite_vertices: list of ndarrays of floats
        For each ridge in `ridge_vertices` the coordinates of the far-point, if it is an infinite ridge.
    infinite_regions: list of list of ints
        List of Voronoi regions with infinite ridges. If positive, the indices are indices to `vertices`.
        If negative they are indices into `infinite_vertices` (`-index-1`).
    
    Convex hull
    -----------
    hull_points: list of ints
        List of indices of the points in `points` making up the convex hull.
    hull_center: ndarray of floats
        Center of mass of the points making up the convex hull.
    inside_vertices: ndarray of boolean
        Indicates for each vertex in `vertices` whether it is inside the convex hull.
    in_hull(): Test if points are within the convex hull of the input points.
    hull_area(): The area contained in the convex hull.
    
    Outer hull
    ----------
    The outer hull is constructed from the convex hull by expanding each
    point of the convex hull away from the center of mass of the convex
    hull such that randomly placing the same number of points in this
    area results on average in the same mean nearest-neighbor
    distance. This enlarged hull is needed for bootstrapping. Note that
    in some cases the outer hull can be *smaller* than the convex hull.
    
    outer_hull_points: 2-d ndarray of floats
        Array of coordinates of the points of the outer hull.
    outer_min_bound: ndarray of floats
        Lower left corner of the bounding-box of the outer hull.
    outer_max_bound: ndarray of floats
        Upper right corner of the bounding-box of the outer hull.
    in_outer_hull(): Test if points are within the outer hull.
    outer_hull_area(): The area contained in the outer hull.
    
    Bootstrap Voronoi diagrams
    --------------------------
    random_points(): Generate random points within the area defined by the input points.

    Plotting the Voronoi diagram
    ----------------------------
    plot_points(): Plot and optionally annotate the input points of the Voronoi diagram.
    plot_center(): Plot the center of mass of the input points.
    plot_vertices(): Plot and optionally annotate the vertices of the Voronoi diagram.
    plot_distances(): Plot lines connecting the neighbors in the Voronoi diagram.
    plot_ridges(): Plot the finite ridges of the Voronoi diagram.
    plot_infinite_ridges(): Plot the infinite ridges of the Voronoi diagram.
    fill_regions(): Fill each finite region of the Voronoi diagram with a color.
    fill_infinite_regions(): Fill each infinite region of the Voronoi diagram with a color.
    
    Plotting the convex hull
    ------------------------
    plot_hull(): Plot the convex hull line containing the input points.
    fill_hull(): Fill the convex hull.
    plot_hull_center(): Plot the center of mass of the convex hull.
    plot_outer_hull(): Plot the outer hull edge.
    fill_outer_hull(): Fill the outer hull.
    
    Usage
    -----
    Generate 20 random points in 2-D:
    ```
    import numpy as np
    points = np.random.rand(20, 2)
    ```
    
    Calculate the Voronoi diagram:
    ```
    from thunderfish.voronoi import Voronoi
    vor = Voronoi(points)
    ```
    
    Retrieve nearest-neighbor distances and compute Voronoi areas:
    ```
    dists = vor.nearest_distances
    areas = vor.areas()
    ```
    
    Generate random points drawn from a Poisson process with the same
    intensity and mean nearest neighbor distance as the data:
    ```
    new_points = vor.random_points(poisson=True, mode='outer')
    ```
    
    Plot Voronoi regions, distances, and input points:
    ```
    import matplotlib.pyplot as plt
    vor.fill_regions(colors=['red', 'green', 'blue', 'orange', 'cyan'], alpha=0.3)
    vor.plot_distances(color='red')
    vor.plot_points(text='p%d', c='c', s=100)
    ```
    """
    
    def __init__(self, points, radius=None, qhull_options=None):
        self.vor = sp.Voronoi(points, furthest_site=False, incremental=False,
                              qhull_options=qhull_options)
        if self.vor.ndim != 2:
            raise ValueError("Only 2D input points are supported.")
        self.hull = sp.Delaunay(points, furthest_site=False, incremental=False,
                                qhull_options=qhull_options)
        self._compute_distances()
        self._compute_infinite_vertices()
        self._compute_hull(qhull_options)
        self.ndim = self.vor.ndim
        self.npoints = self.vor.npoints
        self.points = self.vor.points
        self.vertices = self.vor.vertices
        self.regions = self.vor.regions
        self.ridge_points = self.vor.ridge_points
        self.ridge_vertices = self.vor.ridge_vertices
        self.min_bound = self.vor.min_bound
        self.max_bound = self.vor.max_bound
        self.center = np.mean(self.points, axis=0)

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

        # For each point the distance to its nearest neighbor:
        self.nearest_points = []
        self.nearest_distances = np.zeros(len(self.neighbor_distances))
        for k in range(len(self.neighbor_distances)):
            self.nearest_points.append(self.neighbor_points[k][0])
            self.nearest_distances[k] = self.neighbor_distances[k][0]
        self.mean_nearest_distance = np.mean(self.nearest_distances)
        self.mean_nearest_distance *= 1.0-2.0*np.sqrt(1.0/np.pi-0.25)/np.sqrt(self.vor.npoints)

    def _compute_infinite_vertices(self, radius=None):
        """
        Compute far points of infinite ridges.

        Parameters
        ----------
        radius: float or None
            Radius for computing far points of infinite ridges.
            If None twice the maximum extent of the input points is used.

        Note
        ----
        Code inspired by http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
        """
        # For each ridge, compute far point:
        center = self.vor.points.mean(axis=0)
        if radius is None:
            radius = 2.0*self.vor.points.ptp(axis=0).max()
        self.infinite_vertices = []
        for points, vertices in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            vertices = np.asarray(vertices)
            if np.all(vertices >= 0):
                self.infinite_vertices.append(np.array([]))
            else:
                i = vertices[vertices >= 0][0]  # finite end Voronoi vertex
                t = self.vor.points[points[1]] - self.vor.points[points[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = self.vor.points[points].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.vor.vertices[i] + direction * radius
                self.infinite_vertices.append(far_point)
        # Assemble list of infinite regions:
        # Indices to self.infinite_vertices are negative minus one.
        self.infinite_regions = []
        for rvertices in self.vor.regions:
            if -1 in rvertices:
                new_rvertices = []
                prev_vertex = rvertices[-1]
                # find index of data point enclosed by the region:
                ridge_points = []
                for p, v in zip(self.vor.ridge_points, self.vor.ridge_vertices):
                    if not -1 in v and set(v) <= set(rvertices):
                        ridge_points.extend(p)
                region_point = None
                for rp in ridge_points:
                    if ridge_points.count(rp) > 1:
                        region_point = rp
                        break
                # fill in far points for each region:
                for v_inx, v in enumerate(rvertices):
                    if v >= 0:
                        new_rvertices.append(v)
                    else:
                        for v1_inx, (points, vertices) in enumerate(zip(self.vor.ridge_points,
                                                                        self.vor.ridge_vertices)):
                            if prev_vertex in vertices and -1 in vertices and \
                              (region_point is None or region_point in points):
                                new_rvertices.append(-v1_inx-1)
                                break
                        next_vertex = rvertices[0]
                        if v_inx+1 < len(rvertices):
                            next_vertex = rvertices[v_inx+1]
                        for v2_inx, (points, vertices) in enumerate(zip(self.vor.ridge_points,
                                                                        self.vor.ridge_vertices)):
                            if next_vertex in vertices and -1 in vertices and \
                              (region_point is None or region_point in points) and \
                              new_rvertices[-1] != -v2_inx-1:
                                new_rvertices.append(-v2_inx-1)
                                break
                    prev_vertex = v
                self.infinite_regions.append(new_rvertices)
                
    def _flatten_simplices(self, simplices):
        """
        Transforms list of simplex indices to list of vertex indices.

        In particular, transforms the Delaunay.convex_hull to a list of points
        of the hull that can then be easily plotted.

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

    def _compute_hull(self, qhull_options):
        """
        Compute properties of the convex hull and set up the outer hull.
        """
        self.inside_vertices = self.in_hull(self.vor.vertices)
        self.hull_points = self._flatten_simplices(self.hull.convex_hull)
        self.hull_center = np.mean(self.hull.points[self.hull_points], axis=0)

        # enlarge hull:
        # estimate area needed for a poisson process with same mean distance:
        new_area = 4.0 * self.mean_nearest_distance**2 * self.vor.npoints
        fac = np.sqrt(new_area/self.hull_area())
        self.outer_hull_points = np.zeros((len(self.hull_points), self.vor.ndim))
        for k, point in enumerate(self.hull.points[self.hull_points]):
            point -= self.hull_center
            point *= fac
            point += self.hull_center
            self.outer_hull_points[k] = point
        # compute outer hull:
        self.outer_hull = sp.Delaunay(self.outer_hull_points,
                                      furthest_site=False, incremental=False,
                                      qhull_options=qhull_options)
        self.outer_min_bound = np.min(self.outer_hull_points, axis=0)
        self.outer_max_bound = np.max(self.outer_hull_points, axis=0)
        
    def in_hull(self, p):
        """
        Test if points `p` are within convex hull of the input points.

        Parameters
        ----------
        p: 2-d ndarray
            Array of points to be tested.

        Returns
        -------
        inside: ndarray of booleans
            For each point in `p` whether it is inside the hull.
        """
        inside = self.hull.find_simplex(p) >= 0
        return inside
        
    def in_outer_hull(self, p):
        """
        Test if points `p` are within the outer hull.

        Parameters
        ----------
        p: 2-d ndarray
            Array of points to be tested.

        Returns
        -------
        inside: ndarray of booleans
            For each point in `p` whether it is inside the outer hull.
        """
        inside = self.outer_hull.find_simplex(p) >= 0
        return inside

    def point_types(self):
        """
        The type of the Voronoi regions for each input point.

        Returns
        -------
        points: ndarray of ints
            Indicates the type of Voronoi region associated with each pint in `points`:
            2: finite region with all vertices inside hull,
            1: finite region,
            0: infinite region.
        """
        points = np.zeros(len(self.vor.points), dtype=int) + 2
        for i in range(len(self.vor.points)):
            for j, rp in enumerate(self.vor.ridge_points):
                if i in rp:
                    if not np.all(self.inside_vertices[self.vor.ridge_vertices[j]]) and\
                      points[i] > 0:
                        points[i] = 1
                    if not np.all(np.array(self.vor.ridge_vertices[j]) >= 0) and\
                      points[i] > -1:
                        points[i] = 0
        return points

    def ridge_lengths(self):
        """
        Length of Voronoi ridges between nearest neighbors.

        May be used, for example, as a weigth for `ridge_distances`.

        Returns
        -------
        distances: ndarray of floats
            The length of each ridge in `ridge_vertices`.
            np.inf if one vertex is at infinity.
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

        Returns
        -------
        areas: ndarray of floats
            For each ridge in `ridge_points` or `ridge_vertices`
            its corresponding triangular area.
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

        Parameters
        ----------
        mode: string
            'inside': Calculate area of finite Voronoi regions
                      whose vertices are all inside the hull,
                      set all other to zero.
            'finite_inside': Calculate area of all Voronoi regions.
                    Consider only areas of finite ridges
                    whose vertices are all inside the hull.
            'full': Calculate area of finite Voronoi regions only,
                    set all other to zero.
            'finite': Calculate area of all Voronoi regions. From infinite regions
                    only areas contributed by finite ridges are considered.

        Returns
        -------
        areas: array of floats
            For each point in `points` its corresponding area.
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
                            a = 0.0
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
                            a = 0.0
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
            print('  inside: Finite Voronoi regions whose vertices are all inside the hull.')
            print('  finite_inside: Use all areas corresponding to finite ridges whose vertices are all inside the hull.')
            print('  full: Finite Voronoi regions only.')
            print('  finite: Use all areas corresponding to finite ridges.')
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

    def outer_hull_area(self):
        """
        The area of the outer hull.
        
        Returns
        -------
        area: float
            The area of the outer hull.
        """
        # two sides of the simplex triangles:
        ab = self.outer_hull.points[self.outer_hull.simplices[:,0],:] - self.outer_hull.points[self.outer_hull.simplices[:,1],:]
        cb = self.outer_hull.points[self.outer_hull.simplices[:,2],:] - self.outer_hull.points[self.outer_hull.simplices[:,1],:]
        # area of each simplex is half of the absolute value of the cross product:
        area = 0.5*np.sum(np.abs(np.cross(ab, cb)))
        return area

    def random_points(self, n=None, poisson=True, mode='outer'):
        """
        Generate random points.

        Parameters
        ----------
        n: int or None
            Number of random points to be generated.
            If None `n` is set to the number of input points.
        poisson: boolean
            If `True` then draw the number of points from a Poisson distribution
            with mean number of points given by `n`.
        mode: string
            'bbox' place points randomly in rectangular bounding box of the Voronoi diagram.
            'hull' place points randomly within convex hull of input data.
            'outer' place points randomly within outer hull. The mean nearest-neighbor distance
            between the generated points will be close to the observed one.

        Returns
        -------
        points: 2-D array
            List of randomly generated points.
        """
        # number of points:
        if n is None:
            n = self.npoints
        nn = n
        if poisson:
            nn = np.random.poisson(n)
        m = nn//2
        if m < 5:
            m = 5
        # get bounding box:
        if mode == 'outer':
            min_bound = self.outer_min_bound
            max_bound = self.outer_max_bound
        else:
            min_bound = self.min_bound
            max_bound = self.max_bound
        delta = np.max(max_bound - min_bound)
        points = np.zeros((0, self.ndim))
        while len(points) < nn:
            # random points within bounding box:
            newpoints = np.random.rand(m, self.ndim)
            newpoints *= delta
            newpoints += min_bound
            if mode == 'outer':
                # only take the ones within outer hull:
                inside = self.in_outer_hull(newpoints)
                points = np.vstack((points, newpoints[inside]))
            elif mode == 'hull':
                # only take the ones within hull:
                inside = self.in_hull(newpoints)
                points = np.vstack((points, newpoints[inside]))
            elif mode == 'bbox':
                points = np.vstack((points, newpoints[np.all(newpoints<max_bound, axis=1),:]))
            else:
                print('')
                print('Voronoi.random_points(): unknown value "%s" for the mode parameter:' % mode)
                print('Use one of the following values:')
                print('  bbox: Place points within rectangular bounding box.')
                print('  hull: Place points inside the convex hull.')
                print('  outer: Place points inside the outer hull.')
                print('')
                return
        return points[:nn]

    def plot_points(self, ax=None, text=None, text_offs=(0, 0.05), text_align='center',
                    **kwargs):
        """
        Plot and optionally annotate the input points of the Voronoi diagram.

        Parameters
        ----------
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
        
    def plot_center(self, ax=None, **kwargs):
        """
        Plot the center of mass of the input points.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If `None`, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the `matplotlib.plot()` function.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.center[0], self.center[1], 'o', **kwargs)
        
    def plot_vertices(self, ax=None, text=None, text_offs=(0, 0.05), text_align='center',
                      **kwargs):
        """
        Plot and optionally annotate the vertices of the Voronoi diagram.

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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

    def fill_regions(self, ax=None, inside=None, colors=None, **kwargs):
        """
        Fill each finite region of the Voronoi diagram with a color.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        inside: boolean or None
            True: plot only finite regions with all vertices inside the hull
            False: plot only finite regions with at least one vertex outside the hull
            None: plot all finite regions
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
                    inside_hull = self.in_hull(polygon)
                    if inside is None or (inside and all(inside_hull)) or (not inside and any(inside_hull)):
                        c += 1
                        if colors is None:
                            ax.fill(polygon[:, 0], polygon[:, 1], lw=0, **kwargs)
                        else:
                            ax.fill(polygon[:, 0], polygon[:, 1],
                                    color=colors[c % len(colors)], lw=0, **kwargs)

    def fill_infinite_regions(self, ax=None, colors=None, **kwargs):
        """
        Fill each infinite region of the Voronoi diagram with a color.

        Parameters
        ----------
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
        for region in self.infinite_regions:
            polygon = []
            for p in region:
                if p >= 0:
                    polygon.append(self.vertices[p])
                else:
                    polygon.append(self.infinite_vertices[-p-1])
            if len(polygon) > 0:
                polygon = np.asarray(polygon)
                c += 1
                if colors is None:
                    ax.fill(polygon[:, 0], polygon[:, 1], lw=0, **kwargs)
                else:
                    ax.fill(polygon[:, 0], polygon[:, 1],
                            color=colors[c % len(colors)], lw=0, **kwargs)
        
    def plot_hull(self, ax=None, **kwargs):
        """
        Plot the hull line containing the input points.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.hull.points[self.hull_points, 0],
                self.hull.points[self.hull_points, 1], **kwargs)

    def fill_hull(self, ax=None, **kwargs):
        """
        Fill the hull containing the input points with a color.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.fill() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.fill(self.hull.points[self.hull_points, 0],
                self.hull.points[self.hull_points, 1], lw=0, **kwargs)
        
    def plot_hull_center(self, ax=None, **kwargs):
        """
        Plot the center of mass of the convex hull of the input points.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.hull_center[0], self.hull_center[1], 'o', **kwargs)
        
    def plot_outer_hull(self, ax=None, **kwargs):
        """
        Plot the hull line containing the input points and the vertices of the Voronoi diagram.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.plot() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.outer_hull_points[:, 0],
                self.outer_hull_points[:, 1], **kwargs)

    def fill_outer_hull(self, ax=None, **kwargs):
        """
        Fill the hull containing the input points and the vertices of the Voronoi diagram.

        Parameters
        ----------
        ax: matplotlib.Axes or None
            The axes to be used for plotting. If None, then the current axes is used.
        **kwargs:
            Key-word arguments that are passed on to the matplotlib.fill() function.
        """
        if ax is None:
            ax = plt.gca()
        ax.fill(self.outer_hull_points[:, 0],
                self.outer_hull_points[:, 1], lw=0, **kwargs)

        
if __name__ == "__main__":
    import scipy.stats as st
    import matplotlib.pyplot as plt
    
    print("Checking voronoi module ...")

    # generate random points:
    rs = np.random.randint(0xffffffff)
    #rs = 1718315706  # n=10
    #rs = 803645102   # n=10
    #rs = 2751318392  # double infinite ridges at vertex 0 (n=10)
    #rs = 4226093154 out-hull fac < 1.0
    
    print('random seed: %ld' % rs)
    np.random.seed(rs)
    n = 20    # number of points
    ar = 1.0  # aspect ratio of area in which the points should be placed
    points = np.random.rand(int(n//ar), 2)
    points = points[points[:, 1] < ar, :]
    
    # calculate Voronoi diagram:
    vor = Voronoi(points)
        
    # what we get is:
    print('dimension: %d' % vor.ndim)
    print('number of points: %d' % vor.npoints)
    print('')
    print('distances of nearest neighbors:')
    print(vor.nearest_distances)
    print('for each point its nearest neighbor:')
    print(vor.nearest_points)
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
    for mode in ['inside', 'finite_inside', 'full', 'finite']:
        print('Voronoi area of each point (%s):' % mode)
        print(vor.areas(mode))
    print('Comparison of sum of Voronoi areas (inside < finite_inside < full < finit):')
    a = ['%.2f' % np.sum(vor.areas(mode)) for mode in ['inside', 'finite_inside', 'full', 'finite']]
    print(' < '.join(a))
    print('Type of Voronoi area of each point:')
    print(vor.point_types())

    # plot Voronoi diagram:
    plt.title('Voronoi')
    vor.fill_regions(inside=True, colors=['red', 'green', 'blue', 'orange', 'cyan', 'magenta'], alpha=1.0, zorder=0)
    vor.fill_regions(inside=False, colors=['red', 'green', 'blue', 'orange', 'cyan', 'magenta'], alpha=0.4, zorder=0)
    vor.fill_infinite_regions(colors=['red', 'green', 'blue', 'orange', 'cyan', 'magenta'], alpha=0.1, zorder=0)
    vor.plot_distances(color='red')
    #vor.plot_ridges(c='k', lw=2)
    #vor.plot_infinite_ridges(c='k', lw=2, linestyle='dashed')
    vor.plot_points(text='p%d', c='c', s=100, zorder=10)
    vor.plot_vertices(text='v%d', c='r', s=60)
    ex = 0.3
    plt.xlim(vor.min_bound[0]-ex, vor.max_bound[0]+ex)
    plt.ylim(vor.min_bound[1]-ex, vor.max_bound[1]+ex)
    plt.axes().set_aspect('equal')

    # Convex hull:
    print('Convex hull:')
    print('Area of convex hull: %g' % vor.hull_area())
    print('Area of outer hull: %g' % vor.outer_hull_area())
    new_points = vor.random_points(poisson=True, mode='outer')

    # plot convex hull:
    plt.figure()
    vor.fill_outer_hull(color='black', alpha=0.1)
    vor.plot_outer_hull(color='m', lw=2, label='outer hull')
    vor.fill_hull(color='black', alpha=0.2)
    vor.plot_hull(color='r', lw=2, label='convex hull')
    vor.plot_hull_center(color='r', ms=16)
    vor.plot_hull_center(color='k', ms=4)
    vor.plot_center(color='c', ms=16)
    vor.plot_center(color='k', ms=4)
    vor.plot_points(text='p%d', c='c', s=100, zorder=10, label='input points')
    plt.plot(new_points[:, 0], new_points[:, 1], 'ob', ms=10, label='random points')
    plt.xlim(vor.min_bound[0]-ex, vor.max_bound[0]+ex)
    plt.ylim(vor.min_bound[1]-ex, vor.max_bound[1]+ex)
    plt.axes().set_aspect('equal')
    plt.legend()
        
    # bootstrap:
    def bootstrapped_nearest_distances(vor, n, poisson, mode, ax1, ax2, ax3, bins):
        ps = 'fixed'
        if poisson:
            ps = 'poisson'
        print('bootstrap %s %s ...' % (mode, ps))
        hist_distances = np.zeros((n, len(bins)-1))
        mean_distances = []
        cvs = []
        for j in range(n):
            points = vor.random_points(poisson=poisson, mode=mode)
            if len(points) < 4:
                continue
            try:
                bvor = Voronoi(points)
            except:
                continue
            hist_distances[j,:] = np.histogram(bvor.nearest_distances, bins=bins, normed=True)[0]
            mean_distances.append(np.mean(bvor.nearest_distances))
            cvs.append(np.std(bvor.nearest_distances)/np.mean(bvor.nearest_distances))
        ax1.set_title('bootstrap %s %s' % (mode, ps))
        ax1.hist(mean_distances, 30, normed=True, label='bootstrap')
        de = np.mean(vor.nearest_distances)
        sem = 2.1*0.26136*np.sqrt(vor.outer_hull_area())/vor.npoints # note factor 2.1!
        h = 0.4/sem
        x = np.linspace(de-4.0*sem, de+4.0*sem, 200)
        p = st.norm.pdf(x, loc=de, scale=sem)
        ax1.plot(x, p, 'r', lw=2, label='Gaussian')
        ax1.plot([de, de], [0.0, h], 'r', lw=4, label='observed a.n.n.')
        bmd = np.mean(mean_distances)
        ax1.plot([bmd, bmd], [0.0, h], 'g', lw=4, label='bootstrapped a.n.n.')
        ax1.set_xlabel('mean nearest-neighbor distance')
        ax1.legend()
        ax2.set_title('bootstrap %s %s' % (mode, ps))
        ax2.hist(cvs, 30, normed=True, label='bootstrap')
        # observed cv:
        cvo = np.std(vor.nearest_distances)/np.mean(vor.nearest_distances)
        # significance from bootstrap:
        pb = 0.01*st.percentileofscore(cvs, cvo)
        if pb > 0.5:
            pb = 1.0-pb
        pb *= 2.0
        # significance from z-score:
        secv = 2.0*np.sqrt(1.0/np.pi-0.25)/np.sqrt(vor.npoints)
        zcv = (cvo - 2.0*0.26136)/secv
        pz = 2.0*(1.0 - st.norm.cdf(np.abs(zcv)))
        h = 0.4/secv
        ax2.plot([cvo, cvo], [0.0, h], 'r', lw=4, label=r'observed CV $\alpha=$%.0f%%' % (100.0*pz))
        cvb = np.mean(cvs)
        x = np.linspace(cvb-4.0*secv, cvb+4.0*secv, 200)
        p = st.norm.pdf(x, loc=cvb, scale=secv)
        ax2.plot(x, p, 'g', lw=2, label='Gaussian')
        ax2.plot([cvb, cvb], [0.0, h], 'g', lw=4, label=r'bootstrapped CV $\alpha=$%.0f%%' % (100.0*pb))
        ax2.set_xlabel('CV')
        ax2.legend()
        # nearest-neighbor distance histogram:
        ax3.set_title('bootstrap %s %s' % (mode, ps))
        ax3.fill_between(bins[:-1], *np.percentile(hist_distances, [2.5, 97.5], axis=0), facecolor='blue', alpha=0.5)
        ax3.plot(bins[:-1], np.median(hist_distances, axis=0), 'b', lw=2)
        hd = np.histogram(vor.nearest_distances, bins=bins, normed=True)[0]
        ax3.plot(bins[:-1], hd, 'r', lw=2)
        ax3.set_xlim(0.0, 0.5)
        ax3.set_xlabel('nearest-neighbor distance')
        
    print('Mean distance: %g' % np.mean(vor.nearest_distances))
    print('CV: %g' % (np.std(vor.nearest_distances)/np.mean(vor.nearest_distances)))
    bins = np.linspace(0.0, 1.0, 30)
    nb = 300
    fig1 = plt.figure()
    ax11 = ax1 = fig1.add_subplot(2, 2, 1)
    fig2 = plt.figure()
    ax21 = ax2 = fig2.add_subplot(2, 2, 1)
    fig3 = plt.figure()
    ax31 = ax3 = fig3.add_subplot(2, 2, 1)
    k = 1
    for poisson in [False, True]:
        for mode in ['hull', 'outer']:
            if k > 1:
                ax1 = fig1.add_subplot(2, 2, k, sharex=ax11)
                ax2 = fig2.add_subplot(2, 2, k, sharex=ax21)
                ax3 = fig3.add_subplot(2, 2, k, sharex=ax31)
            bootstrapped_nearest_distances(vor, nb, poisson, mode, ax1, ax2, ax3, bins)
            k += 1
    for f in [fig1, fig2, fig3]:
        f.tight_layout()
    print('... done.')
    plt.tight_layout()
    plt.show()

