from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import matplotlib.pyplot as plt
import thunderfish.fishshapes as fs


def test_plotfish():
    bodykwargs=dict(lw=1, edgecolor='k', facecolor='k')
    finkwargs=dict(lw=1, edgecolor='k', facecolor='grey')
    eyekwargs=dict(lw=1, edgecolor='white', facecolor='grey')
    for k, fish in fs.fish_shapes.items():
        for j in range(10):
            fish_coords = (np.random.randn(2)*10.0, (0, 0),
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            xx, yy, zz = fs.fish_surface(fish, *fish_coords, gamma=np.random.rand(1)*2+0.1)
            nx, ny, nz = fs.surface_normals(xx, yy, zz)
            fish_coords = (np.random.randn(2)*10.0, np.random.rand(2)-0.5,
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            fig, ax = plt.subplots()
            fs.plot_fish(ax, fish, *fish_coords, bodykwargs=bodykwargs, finkwargs=finkwargs,
                         eyekwargs=eyekwargs)
            plt.close()
    for k in fs.fish_shapes.keys():
        for j in range(10):
            fish_coords = (np.random.randn(2)*10.0, np.random.rand(2)-0.5,
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            fig, ax = plt.subplots()
            fs.plot_fish(ax, k, *fish_coords, bodykwargs=bodykwargs, finkwargs=finkwargs,
                         eyekwargs=eyekwargs)
            plt.close()
    for k, fish in fs.fish_top_shapes.items():
        for j in range(10):
            fish_coords = (np.random.randn(2)*10.0, np.random.rand(2)-0.5,
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            fig, ax = plt.subplots()
            fs.plot_fish(ax, fish, *fish_coords, bodykwargs=bodykwargs, finkwargs=finkwargs,
                         eyekwargs=eyekwargs)
            plt.close()
    for k in fs.fish_top_shapes.keys():
        for j in range(10):
            fish_coords = (np.random.randn(2)*10.0, np.random.rand(2)-0.5,
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            fig, ax = plt.subplots()
            fs.plot_fish(ax, (k, 'top'), *fish_coords, bodykwargs=bodykwargs,
                         finkwargs=finkwargs, eyekwargs=eyekwargs)
            plt.close()
    for k, fish in fs.fish_side_shapes.items():
        for j in range(10):
            fish_coords = (np.random.randn(2)*10.0, np.random.rand(2)-0.5,
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            fig, ax = plt.subplots()
            fs.plot_fish(ax, fish, *fish_coords, bodykwargs=bodykwargs, finkwargs=finkwargs,
                         eyekwargs=eyekwargs)
            plt.close()
    for k in fs.fish_side_shapes.keys():
        for j in range(10):
            fish_coords = (np.random.randn(2)*10.0, np.random.rand(2)-0.5,
                           np.random.rand(1)*10.0+0.5, (np.random.randn(1)-0.5)*40.0)
            fig, ax = plt.subplots()
            fs.plot_fish(ax, (k, 'side'), *fish_coords, bodykwargs=bodykwargs,
                         finkwargs=finkwargs, eyekwargs=eyekwargs)
            plt.close()
    for j in range(10):
        fig, ax = plt.subplots()
        fs.plot_object(ax, pos=np.random.randn(2)*5, radius=np.random.rand(1)*5.0+0.1)
        plt.close()


def test_extract_fish():
    data = "m 84.013672,21.597656 0.0082,83.002434 0.113201,-0.0145 0.1238,-0.32544 0.06532,-0.80506 0.06836,-0.87696 0.0332,-4.298823 v -8.625 l 0.06836,-1.724609 0.06836,-1.722657 0.07032,-1.726562 0.06836,-1.726563 0.06641,-1.693359 0.03439,-1.293583 0.06912,-1.30798 0.10547,-1.724609 0.10156,-1.724609 0.10352,-1.726563 0.10352,-1.724609 0.13867,-1.72461 0.171876,-2.572265 0.13672,-1.72461 0.13672,-1.726562 0.10352,-1.724609 0.06836,-1.722657 0.103515,-2.574219 0.06836,-1.722656 0.10352,-1.728515 0.07032,-1.722657 0.06836,-1.724609 0.240234,-1.724609 0.34375,-1.72461 0.134766,-1.726562 0.10352,-1.69336 0.03516,-0.875 0.07031,-1.728515 v -0.847657 l -0.07273,-2.246267 -0.0172,-0.184338 0.15636,0.09441 0.384252,1.019739 0.748821,0.905562 1.028854,0.647532 1.356377,-0.03149 0.362644,-0.347764 -0.264138,-0.736289 -1.268298,-1.126614 -1.363988,-0.922373 -0.927443,-0.451153 -0.228986,-0.07018 -0.0015,-0.21624 0.03663,-0.660713 0.480469,-0.847657 -0.101563,-0.876953 -0.103515,-0.845703 -0.103516,-0.876953 -0.207031,-1.695313 -0.273438,-1.724609 -0.308594,-1.726562 -0.27539,-1.72461 -0.310547,-1.722656 -0.240234,-0.878906 -0.400196,-0.877344 -0.53927,-0.596268 -0.486573,-0.216683 z"
    verts = fs.extract_path(data)
    # look at the path:
    fig, ax = plt.subplots()
    fs.plot_pathes(ax, verts)
    ax.set_aspect('equal')
    plt.close()
    # fix path:
    fs.center_pathes(verts)
    fs.flipx_pathes(verts)
    fs.flipy_pathes(verts)
    fs.rotate_pathes(-90.0, verts)
    verts[:,1] *= 0.8               # change aspect ratio
    verts = verts[1:,:]             # remove first point
    fs.translate_pathes(0.0, -np.min(verts[:,1]), verts)
    # mirror, normalize and export path:
    verts = fs.mirror_path(verts)
    fs.normalize_path(verts)
    fish = fs.export_fish('Alepto_top', verts)
    
        
