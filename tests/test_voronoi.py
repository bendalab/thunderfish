from nose.tools import assert_true, assert_equal, assert_raises
import os
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.voronoi import Voronoi

def test_voronoi_dimensions():
    points = np.random.rand(20, 3)
    assert_raises(ValueError, Voronoi, points)
    
    points = np.random.rand(20, 1)
    assert_raises(ValueError, Voronoi, points)


def test_voronoi():
    # generate random points:
    n = 20
    points = np.random.rand(n, 2)

    # calculate Voronoi diagram:
    vor = Voronoi(points)

    # test output:
    assert_equal(vor.npoints, n, 'wrong number of points')
    assert_equal(vor.ndim, 2, 'wrong dimension')

    assert_equal(len(vor.nearest_distances), n, 'wrong len of nearest_distances')
    assert_equal(len(vor.neighbor_distances), n, 'wrong len of neighbor_distances')
    assert_equal(len(vor.neighbor_points), n, 'wrong len of neighbor_points')

    assert_equal(len(vor.ridge_distances), len(vor.ridge_points), 'wrong len of ridge_distances')
    assert_equal(len(vor.ridge_distances), len(vor.ridge_vertices), 'wrong len of ridge_distances')
    assert_equal(len(vor.ridge_lengths()), len(vor.ridge_vertices), 'wrong len of ridge_length()')
    assert_equal(len(vor.ridge_areas()), len(vor.ridge_vertices), 'wrong len of ridge_areas()')
    for mode in ['inside', 'finite_inside', 'full', 'finite']:
        assert_equal(len(vor.areas(mode)), len(vor.points), 'wrong len of ridge_areas()')
    assert_equal(len(vor.point_types()), len(vor.points), 'wrong len of point_types()')
    
    assert_true(np.all(vor.in_hull(points)), 'in_hull() for input points failed')
    assert_true(np.all(vor.in_outer_hull(points)), 'in_outer_hull() for input points failed')

    assert_true(vor.hull_area() <= 1.0, 'hull_area() too large')
    assert_true(vor.hull_area() <= vor.outer_hull_area(), 'outer_hull_area() too large')

    for mode in ['bbox', 'hull', 'outer']:
        new_points = vor.random_points(poisson=False, mode=mode)
        assert_equal(len(new_points), len(vor.points), 'wrong number of points generated in random_points()')
    for mode in ['bbox', 'hull', 'outer']:
        new_points = vor.random_points(poisson=True, mode=mode)
        assert_true(len(new_points) < 2*len(vor.points), 'wrong number of points generated in random_points()')
        

def test_plot_voronoi():
    # generate random points:
    n = 20
    points = np.random.rand(n, 2)

    # calculate Voronoi diagram:
    vor = Voronoi(points)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    vor.fill_regions(ax, inside=True, colors=['red', 'green', 'blue', 'orange', 'cyan', 'magenta'], alpha=1.0, zorder=0)
    vor.fill_regions(ax, inside=False, colors=['red', 'green', 'blue', 'orange', 'cyan', 'magenta'], alpha=0.4, zorder=0)
    vor.fill_infinite_regions(ax, colors=['red', 'green', 'blue', 'orange', 'cyan', 'magenta'], alpha=0.1, zorder=0)
    vor.plot_distances(ax, color='red')
    vor.plot_ridges(ax, c='k', lw=2)
    vor.plot_infinite_ridges(ax, c='k', lw=2, linestyle='dashed')
    vor.plot_points(ax, text='p%d', c='c', s=100, zorder=10)
    vor.plot_vertices(ax, text='v%d', c='r', s=60)

    vor.fill_outer_hull(ax, color='black', alpha=0.1)
    vor.plot_outer_hull(ax, color='m', lw=2, label='outer hull')
    vor.fill_hull(ax, color='black', alpha=0.2)
    vor.plot_hull(ax, color='r', lw=2, label='convex hull')
    vor.plot_hull_center(ax, color='r', ms=16)
    vor.plot_center(ax, color='c', ms=16)

    fig.savefig('test.png')
    assert_true(os.path.exists('test.png'), 'plotting failed')
    os.remove('test.png')
