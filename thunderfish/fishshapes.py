"""
Plot fish outlines.

- `plot_fish()`: plot outline of an electric fish.

For importing fish outlines use the following functions:
- `extract_fish()`: convert SVG coordinates to numpy array.
- `bbox_pathes()`: common bounding box of pathes.
- `plot_pathes()`: plot pathes.
- `translate_pathes()`: translate pathes in place.
- `center_pathes()`: translate pathes to their common origin in place.
- `rotate_pathes()`: rotate pathes in place.
- `mirror_fish()`: complete path of half a fish outline by appending the mirrored path.
- `normalize_fish()`: normalize fish outline to unit length.
- `export_path()`: print coordinates of path for import as numpy array.
- `export_fish()`: print coordinates of fish outlines in dictionary for import.
- `export_fish_demo(): code demonstrating how to export fish outlines from SVG.
"""

from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


def extract_fish(data):
    """ Convert SVG coordinates to numpy array.

    Draw a fish outline in inkscape. Open the XML Editor (shift+ctrl+x)
    and copy the value of the data field ('d') into a variable that you
    pass to this function.
    Alternatively, try the 'inkscape:original-d' variable.

    Parameters
    ----------
    data: string
        Space separated coordinate pairs describing the outline of a fish.
        The coordinates are separated by commas. Coordinate pairs without a comma are ignored.

    Returns
    -------
    vertices: 2D array
        The coordinates of the outline of a fish.
    """
    coords = data.split(' ')
    vertices = []
    relative = False
    xc = yc = 0
    for c in coords:
        if ',' in c:
            xs, ys = c.split(',')
            x = float(xs)
            y = float(ys)
            if relative:
                xc += x
                yc += y
            else:
                xc = x
                yc = y
            vertices.append((xc, yc))
        else:
            if c in 'MLC':
                relative = False
            elif c in 'mlc':
                relative = True
    vertices = np.array(vertices)
    return vertices


def bbox_pathes(*vertices):
    """ Common bounding box of pathes.

    Parameters
    ----------
    vertices: one or more 2D arrays
        The coordinates of pathes.

    Returns
    -------
    bbox: 2D array
        Bounding box of the pathes: [[x0, y0], [x1, y1]]
    """
    # get bounding box of all pathes:
    bbox = np.zeros((2, 2))
    first = True
    for verts in vertices:
        vbbox = np.array([[np.min(verts[:,0]), np.min(verts[:,1])],
                          [np.max(verts[:,0]), np.max(verts[:,1])]])
        if first:
            bbox = vbbox
            first = False
        else:
            bbox[0,0] = min(bbox[0,0], vbbox[0,0])
            bbox[0,1] = min(bbox[0,1], vbbox[0,1])
            bbox[1,0] = max(bbox[1,0], vbbox[1,0])
            bbox[1,1] = max(bbox[1,1], vbbox[1,1])
    return bbox


def plot_pathes(ax, *vertices, **kwargs):
    """ Plot pathes.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the fish.
    vertices: 2D array
        The coordinates of pathes to be plotted.
    kwargs: key word arguments
        Arguments for PathPatch used to draw the path.
    """
    for verts in vertices:
        codes = np.zeros(len(verts))
        codes[:] = Path.LINETO
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(verts, codes)
        ax.add_patch(PathPatch(path, **kwargs))
    bbox = bbox_pathes(*vertices)
    center = np.mean(bbox, axis=0)
    bbox -= center
    bbox *= 1.2
    bbox += center
    ax.set_xlim(*bbox[:,0])
    ax.set_ylim(*bbox[:,1])


def translate_pathes(dx, dy, *vertices):
    """ Translate pathes in place.

    Parameters
    ----------
    dx: float
        Shift in x direction.
    dy: float
        Shift in y direction.
    vertices: one or more 2D arrays
        The coordinates of pathes to be translated.
    """
    for verts in vertices:
        verts[:,0] += dx
        verts[:,1] += dy


def center_pathes(*vertices):
    """ Translate pathes to their common origin in place.

    Parameters
    ----------
    vertices: one or more 2D arrays
        The coordinates of pathes to be centered.
    """
    center = np.mean(bbox_pathes(*vertices), axis=1)
    # shift:
    for verts in vertices:
        verts[:,0] -= center[0]
        verts[:,1] -= center[1]


def rotate_pathes(theta, *vertices):
    """ Rotate pathes in place.

    Parameters
    ----------
    theta: float
        Rotation angle in degrees.
    vertices: one or more 2D arrays
        The coordinates of pathes to be rotated.
    """
    theta *= np.pi/180.0
    # rotation matrix:
    c = np.cos(theta)
    s = np.sin(theta)
    rm = np.array(((c, -s), (s, c)))
    # rotation:
    for verts in vertices:
        verts[:,:] = np.dot(verts, rm)


def mirror_fish(vertices1):
    """ Complete path of half a fish outline by appending the mirrored path.

    It is sufficient to draw half of a top view of a fish. Import with
    extract_fish() and use this function to add the missing half of the
    outline to the path. The outline is mirrored on the x-axis.

    Parameters
    ----------
    vertices1: 2D array
        The coordinates of one half of the outline of a fish.

    Returns
    -------
    vertices: 2D array
        The coordinates of the complete outline of a fish.
    """
    vertices2 = np.array(vertices1[::-1,:])
    vertices2[:,1] *= -1
    vertices = np.concatenate((vertices1, vertices2))
    return vertices


def normalize_fish(midpoint, *vertices):
    """ Normalize and shift fish outline in place.

    The fish extend in x direction is normalized to one and its
    center defined by midpoint is shifted to the origin.

    Parameters
    ----------
    midpoint: float
        A number between 0 and 1 to which the center of the fish is shifted.
    vertices: one or more 2D arrays
        The coordinates of the outline of a fish.
    """
    bbox = bbox_pathes(*vertices)
    # normalize:
    for verts in vertices:
        verts[:,1] -= np.mean(bbox[:,1])
        verts[:,0] -= bbox[0,0]
        verts /= bbox[1,0] - bbox[0,0]
        verts[:,0] -= midpoint


def export_path(vertices):
    """ Print coordinates of path for import as numpy array.

    The variable name, a leading 'np.array([' and the closing '])'
    are not printed.

    Parameters
    ----------
    vertices: 2D array
        The coordinates of the path.
    """
    n = 2
    for k, v in enumerate(vertices):
        if k%n == 0:
            print('   ', end='')
        print(' [%.8e, %.8e],' % (v[0], v[1]), end='')
        if k%n == n-1 and k < len(vertices)-1:
            print('')


def export_fish(name, body, *fins):
    """ Print coordinates of fish outlines in dictionary for import.

    Writes a dictionary with name 'name' and keys 'body', 'fin0', 'fin1', ...
    holding the pathes.

    Copy these coordinates from the console and paste them into this module.
    Give it a proper name and don't forget to add it to the fish_shapes dictionary
    to make it know to plot_fish().

    Parameters
    ----------
    name: string
        Name of the variable.
    body: 2D array
        The coordinates of fish's body.
    fins: zero or more 2D array
        The coordinates of the fish's fins.

    Returns
    -------
    fish: dict
        A dictionary holding the pathes that can be passed directly to plot_fish().
    """
    print('%s = dict(body=np.array([' % name)
    export_path(body)
    fish = dict(body=body)
    for k, f in enumerate(fins):
        print(']),')
        print('    fin%d=np.array([' % k)
        export_path(f)
        fish['fin%d' % k] = f
    print(']))')
    return fish


def export_fish_demo():
    """ Code demonstrating how to export a fish outline from SVG.
    """
    data = "m 84.013672,21.597656 0.0082,83.002434 0.113201,-0.0145 0.1238,-0.32544 0.06532,-0.80506 0.06836,-0.87696 0.0332,-4.298823 v -8.625 l 0.06836,-1.724609 0.06836,-1.722657 0.07032,-1.726562 0.06836,-1.726563 0.06641,-1.693359 0.03439,-1.293583 0.06912,-1.30798 0.10547,-1.724609 0.10156,-1.724609 0.10352,-1.726563 0.10352,-1.724609 0.13867,-1.72461 0.171876,-2.572265 0.13672,-1.72461 0.13672,-1.726562 0.10352,-1.724609 0.06836,-1.722657 0.103515,-2.574219 0.06836,-1.722656 0.10352,-1.728515 0.07032,-1.722657 0.06836,-1.724609 0.240234,-1.724609 0.34375,-1.72461 0.134766,-1.726562 0.10352,-1.69336 0.03516,-0.875 0.07031,-1.728515 v -0.847657 l -0.07273,-2.246267 -0.0172,-0.184338 0.15636,0.09441 0.384252,1.019739 0.748821,0.905562 1.028854,0.647532 1.356377,-0.03149 0.362644,-0.347764 -0.264138,-0.736289 -1.268298,-1.126614 -1.363988,-0.922373 -0.927443,-0.451153 -0.228986,-0.07018 -0.0015,-0.21624 0.03663,-0.660713 0.480469,-0.847657 -0.101563,-0.876953 -0.103515,-0.845703 -0.103516,-0.876953 -0.207031,-1.695313 -0.273438,-1.724609 -0.308594,-1.726562 -0.27539,-1.72461 -0.310547,-1.722656 -0.240234,-0.878906 -0.400196,-0.877344 -0.53927,-0.596268 -0.486573,-0.216683 z"
    verts = extract_fish(data)
    # look at the path:
    fig, ax = plt.subplots()
    plot_pathes(ax, verts)
    ax.set_aspect('equal')
    plt.show()
    # fix path:
    center_pathes(verts)
    rotate_pathes(-90.0, verts)
    verts[:,1] *= 0.8               # change aspect ratio
    verts = verts[1:,:]             # remove first point
    translate_pathes(0.0, -np.min(verts[:,1]), verts)
    # mirrow, normalize and export path:
    verts = mirror_fish(verts)
    normalize_fish(0.45, verts)
    fish = export_fish('Alepto_top', verts)
    # plot outline:
    fig, ax = plt.subplots()
    plot_fish(ax, fish, size=-2.0*np.min(verts[:,0])/1.05,
              bodykwargs=dict(lw=1, edgecolor='k', facecolor='r'),
              finkwargs=dict(lw=1, edgecolor='k', facecolor='b'))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1)
    plt.show()


""" Outline of an Apteronotus viewd from top, modified from Krahe 2004. """
Alepto_top = dict(body=np.array([
    [-4.50000000e-01, 0.00000000e+00], [-4.49802704e-01, 1.23222860e-03],
    [-4.45374557e-01, 2.57983066e-03], [-4.34420392e-01, 3.29085947e-03],
    [-4.22487909e-01, 4.03497963e-03], [-3.63995354e-01, 4.39637211e-03],
    [-3.40529212e-01, 5.14049228e-03], [-3.17089631e-01, 5.88461244e-03],
    [-2.93596916e-01, 6.65006783e-03], [-2.70104187e-01, 7.39418800e-03],
    [-2.47063253e-01, 8.11708180e-03], [-2.29461930e-01, 8.49142780e-03],
    [-2.11664711e-01, 9.24382081e-03], [-1.88198570e-01, 1.03918950e-02],
    [-1.64732428e-01, 1.14974077e-02], [-1.41239699e-01, 1.26242555e-02],
    [-1.17773558e-01, 1.37511034e-02], [-9.43074032e-02, 1.52605701e-02],
    [-5.93075081e-02, 1.71314946e-02], [-3.58413531e-02, 1.86197349e-02],
    [-1.23486380e-02, 2.01079753e-02], [1.11175034e-02, 2.12348231e-02],
    [3.45570845e-02, 2.19789433e-02], [6.95835670e-02, 2.31057367e-02],
    [9.30231346e-02, 2.38498569e-02], [1.16542423e-01, 2.49767047e-02],
    [1.39982005e-01, 2.57421601e-02], [1.63448146e-01, 2.64862803e-02],
    [1.86914287e-01, 2.91013032e-02], [2.10380442e-01, 3.28431304e-02],
    [2.33873157e-01, 3.43101008e-02], [2.56914105e-01, 3.54369487e-02],
    [2.68819919e-01, 3.58196764e-02], [2.92339207e-01, 3.65850229e-02],
    [3.22903364e-01, 3.57933339e-02], [3.25411585e-01, 3.56061065e-02],
    [3.24126982e-01, 3.73081344e-02], [3.10251756e-01, 4.14908387e-02],
    [2.97930096e-01, 4.96419915e-02], [2.89119358e-01, 6.08413919e-02],
    [2.89547832e-01, 7.56059835e-02], [2.94279733e-01, 7.95534778e-02],
    [3.04298155e-01, 7.66782524e-02], [3.19627591e-01, 6.28724285e-02],
    [3.32177993e-01, 4.80249888e-02], [3.38316671e-01, 3.79294791e-02],
    [3.39271585e-01, 3.54368942e-02], [3.42213886e-01, 3.54205663e-02],
    [3.51203973e-01, 3.58192954e-02], [3.62737740e-01, 4.10493520e-02],
    [3.74670128e-01, 3.99438067e-02], [3.86177308e-01, 3.88170133e-02],
    [3.98109696e-01, 3.76902090e-02], [4.21177217e-01, 3.54366112e-02],
    [4.44643358e-01, 3.24601523e-02], [4.68136073e-01, 2.91010093e-02],
    [4.91602228e-01, 2.61033022e-02], [5.15041796e-01, 2.27229002e-02],
    [5.27000757e-01, 2.01078773e-02], [5.38938465e-01, 1.57516176e-02],
    [5.47051671e-01, 9.88149348e-03], [5.50000000e-01, 4.58499286e-03],
    [5.50000000e-01, -4.58499286e-03], [5.47051671e-01, -9.88149348e-03],
    [5.38938465e-01, -1.57516176e-02], [5.27000757e-01, -2.01078773e-02],
    [5.15041796e-01, -2.27229002e-02], [4.91602228e-01, -2.61033022e-02],
    [4.68136073e-01, -2.91010093e-02], [4.44643358e-01, -3.24601523e-02],
    [4.21177217e-01, -3.54366112e-02], [3.98109696e-01, -3.76902090e-02],
    [3.86177308e-01, -3.88170133e-02], [3.74670128e-01, -3.99438067e-02],
    [3.62737740e-01, -4.10493520e-02], [3.51203973e-01, -3.58192954e-02],
    [3.42213886e-01, -3.54205663e-02], [3.39271585e-01, -3.54368942e-02],
    [3.38316671e-01, -3.79294791e-02], [3.32177993e-01, -4.80249888e-02],
    [3.19627591e-01, -6.28724285e-02], [3.04298155e-01, -7.66782524e-02],
    [2.94279733e-01, -7.95534778e-02], [2.89547832e-01, -7.56059835e-02],
    [2.89119358e-01, -6.08413919e-02], [2.97930096e-01, -4.96419915e-02],
    [3.10251756e-01, -4.14908387e-02], [3.24126982e-01, -3.73081344e-02],
    [3.25411585e-01, -3.56061065e-02], [3.22903364e-01, -3.57933339e-02],
    [2.92339207e-01, -3.65850229e-02], [2.68819919e-01, -3.58196764e-02],
    [2.56914105e-01, -3.54369487e-02], [2.33873157e-01, -3.43101008e-02],
    [2.10380442e-01, -3.28431304e-02], [1.86914287e-01, -2.91013032e-02],
    [1.63448146e-01, -2.64862803e-02], [1.39982005e-01, -2.57421601e-02],
    [1.16542423e-01, -2.49767047e-02], [9.30231346e-02, -2.38498569e-02],
    [6.95835670e-02, -2.31057367e-02], [3.45570845e-02, -2.19789433e-02],
    [1.11175034e-02, -2.12348231e-02], [-1.23486380e-02, -2.01079753e-02],
    [-3.58413531e-02, -1.86197349e-02], [-5.93075081e-02, -1.71314946e-02],
    [-9.43074032e-02, -1.52605701e-02], [-1.17773558e-01, -1.37511034e-02],
    [-1.41239699e-01, -1.26242555e-02], [-1.64732428e-01, -1.14974077e-02],
    [-1.88198570e-01, -1.03918950e-02], [-2.11664711e-01, -9.24382081e-03],
    [-2.29461930e-01, -8.49142780e-03], [-2.47063253e-01, -8.11708180e-03],
    [-2.70104187e-01, -7.39418800e-03], [-2.93596916e-01, -6.65006783e-03],
    [-3.17089631e-01, -5.88461244e-03], [-3.40529212e-01, -5.14049228e-03],
    [-3.63995354e-01, -4.39637211e-03], [-4.22487909e-01, -4.03497963e-03],
    [-4.34420392e-01, -3.29085947e-03], [-4.45374557e-01, -2.57983066e-03],
    [-4.49802704e-01, -1.23222860e-03], [-4.50000000e-01, -0.00000000e+00],]))

Alepto_male_side = dict(body=np.array([
    [3.30332097e-01, 5.66361974e-02], [2.91127905e-01, 6.08460339e-02],
    [2.41463866e-01, 6.37667812e-02], [1.87379023e-01, 6.32716007e-02],
    [1.19123434e-01, 5.87953634e-02], [3.63948412e-02, 4.89838394e-02],
    [-2.55221954e-02, 3.79211033e-02], [-1.10157310e-01, 2.60651116e-02],
    [-1.82035003e-01, 1.70421484e-02], [-2.49079447e-01, 1.12096081e-02],
    [-3.12251791e-01, 7.77265717e-03], [-3.70527920e-01, 5.72449035e-03],
    [-4.22735573e-01, 6.89606722e-03], [-4.30154179e-01, 7.36398216e-03],
    [-4.42605065e-01, 1.16411701e-02], [-4.47402289e-01, 7.41543089e-03],
    [-4.50000000e-01, -1.34973487e-03], [-4.47832769e-01, -1.02981288e-02],
    [-4.43106950e-01, -1.28380198e-02], [-4.31164618e-01, -6.69215833e-03],
    [-4.22578673e-01, -4.67623978e-03], [-3.95390092e-01, -4.46123207e-03],
    [-3.24805165e-01, -7.55994875e-03], [-2.83716813e-01, -9.33171407e-03],
    [-2.58099380e-01, -1.00017062e-02], [-2.32451613e-01, -1.15396175e-02],
    [-1.84498580e-01, -2.06834039e-02], [-1.36892658e-01, -3.11727999e-02],
    [-5.87387324e-02, -4.84024272e-02], [1.49246121e-02, -5.79218881e-02],
    [8.28767168e-02, -6.43397525e-02], [1.75319086e-01, -7.06513967e-02],
    [2.49523049e-01, -7.84740377e-02], [2.87035792e-01, -8.29828746e-02],
    [3.24475366e-01, -8.53964222e-02], [3.62742824e-01, -8.19038538e-02],
    [3.86340505e-01, -7.67231052e-02], [4.05492327e-01, -7.06451372e-02],
    [4.24670470e-01, -6.30564452e-02], [4.32920881e-01, -5.91824740e-02],
    [4.34828678e-01, -5.77550188e-02], [4.36562866e-01, -5.84353292e-02],
    [4.40753372e-01, -5.86589139e-02], [4.53494946e-01, -5.75960624e-02],
    [4.88474761e-01, -5.98270958e-02], [5.11389913e-01, -6.32960653e-02],
    [5.27010163e-01, -6.71433852e-02], [5.34437594e-01, -6.74404376e-02],
    [5.40842798e-01, -6.67840745e-02], [5.44567181e-01, -6.43050992e-02],
    [5.45443985e-01, -6.15972915e-02], [5.44497789e-01, -5.95849672e-02],
    [5.41729699e-01, -5.85016417e-02], [5.34298546e-01, -5.63808423e-02],
    [5.43112897e-01, -5.30550750e-02], [5.47742360e-01, -4.97667864e-02],
    [5.50000000e-01, -4.58196050e-02], [5.49521047e-01, -4.21153641e-02],
    [5.46159278e-01, -3.72756471e-02], [5.36402575e-01, -3.03513600e-02],
    [5.17134496e-01, -1.91920392e-02], [4.89218141e-01, -4.42866758e-03],
    [4.75010402e-01, 5.95359753e-03], [4.64788070e-01, 1.54860523e-02],
    [4.43656086e-01, 2.59160740e-02], [4.25679976e-01, 3.09323720e-02],
    [4.11404254e-01, 3.84002337e-02], [3.87900061e-01, 4.55458302e-02],
    [3.61463577e-01, 5.12553862e-02],]),
    fin0=np.array([
    [3.79593304e-01, -7.80912941e-02], [3.77561074e-01, -8.33367726e-02],
    [3.58709726e-01, -9.75609654e-02], [3.30934315e-01, -1.06562137e-01],
    [3.08017473e-01, -1.11378542e-01], [2.85142157e-01, -1.12967112e-01],
    [2.68081531e-01, -1.10854592e-01], [2.48185626e-01, -1.09221292e-01],
    [2.28099090e-01, -1.12140193e-01], [2.09752865e-01, -1.17262090e-01],
    [1.90752841e-01, -1.18766781e-01], [1.77904629e-01, -1.16212356e-01],
    [1.69134213e-01, -1.10784346e-01], [1.59580014e-01, -1.02936264e-01],
    [1.32018471e-01, -9.45992770e-02], [1.00559867e-01, -9.42289586e-02],
    [7.74790284e-02, -1.02521601e-01], [5.39270492e-02, -1.07334461e-01],
    [1.87289863e-02, -1.07465162e-01], [-8.88654880e-03, -1.02320945e-01],
    [-2.82549598e-02, -9.30428977e-02], [-4.94601687e-02, -8.05174600e-02],
    [-7.99416404e-02, -6.86658117e-02], [-1.08259295e-01, -6.58695624e-02],
    [-1.36001442e-01, -6.86570716e-02], [-1.64339679e-01, -6.64007295e-02],
    [-1.88708971e-01, -5.63982408e-02], [-2.05168178e-01, -4.26230327e-02],
    [-2.21293058e-01, -3.13785159e-02], [-2.38416341e-01, -2.71291801e-02],
    [-2.56103856e-01, -2.67461533e-02], [-2.72345146e-01, -2.32128039e-02],
    [-2.88333410e-01, -1.29124469e-02], [-2.93264223e-01, -8.86918932e-03],
    [-2.58609907e-01, -9.75713562e-03], [-2.36088545e-01, -1.10633718e-02],
    [-2.09977440e-01, -1.50414203e-02], [-1.66119429e-01, -2.49072954e-02],
    [-1.18443229e-01, -3.53996137e-02], [-6.27179441e-02, -4.73585838e-02],
    [-2.07908982e-02, -5.36259998e-02], [3.19093361e-02, -6.01068165e-02],
    [7.75299392e-02, -6.38080982e-02], [1.27139003e-01, -6.70205020e-02],
    [1.71071140e-01, -7.10104673e-02], [2.28723549e-01, -7.70286908e-02],
    [2.82100395e-01, -8.25268651e-02], [3.24938812e-01, -8.59456072e-02],
    [3.60041908e-01, -8.28007219e-02],]))


""" Dictionary holding all known electric fish shapes. """
fish_shapes = dict(Alepto_top=Alepto_top,
                   Alepto_male_side=Alepto_male_side)

""" Dictionary holding all known electric fish shapes viewed from top. """
fish_top_shapes = dict(Alepto=Alepto_top)

""" Dictionary holding all known electric fish shapes viewed from the side. """
fish_side_shapes = dict(Alepto=Alepto_male_side)
    

def plot_fish(ax, fish, pos=(0, 0), direction=(1, 0), size=20.0, bend=0, scaley=1,
              bodykwargs={}, finkwargs={}):
    """ Plot outline of an electric fish.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the fish.
    fish: string or tuple or dict
        Specifies a fish to show:
        - any of the strings defining a shape contained in the `fish_shapes` dictionary,
        - a tuple with the name of the fish and 'top' or 'side',
        - a dictionary with at least a 'body' key holding pathes to be drawn.
    pos: tuple of floats
        Coordinates of the fish's position (its center).
    direction: tuple of floats
        Coordinates of a vector defining the orientation of the fish.
    size: float
        Size of the fish.
    bend: float
        Bending angle of the fish's tail in degree.
    scaley: float
        Scale factor applied in y direction after bending and rotation to
        compensate for differntly scaled axes.
    bodykwargs: dict
        Key-word arguments for PathPatch used to draw the fish's body.
    finkwargs: dict
        Key-word arguments for PathPatch used to draw the fish's fins.
    """
    # retrieve fish shape:
    if not isinstance(fish, dict):
        if isinstance(fish, (tuple, list)):
            if fish[1] == 'top':
                fish = fish_top_shapes[fish[0]]
            else:
                fish = fish_side_shapes[fish[0]]
        else:
            fish = fish_shapes[fish]
    bbox = bbox_pathes(*fish.values())
    size_fac = -1.05*0.5/bbox[0,0]
    for part, verts in fish.items():
        verts = np.array(verts)
        verts *= size_fac*size
        if np.abs(bend) > 1.e-8:
            sel = verts[:,0]<0.0
            xp = verts[sel,0]   # x coordinates of all negative y coordinates of verts
            yp = verts[sel,1]   # all negative y coordinates of verts
            r = -180.0*0.5*size/bend/np.pi              # radius of circle on which to bend the tail
            beta = xp/r                                 # angle on circle for each y coordinate
            verts[sel,0] = -np.abs((r+yp)*np.sin(beta)) # transformed x coordinates
            verts[sel,1] = -r+(r+yp)*np.cos(beta)        # transformed y coordinates
        codes = np.zeros(len(verts))
        codes[:] = Path.LINETO
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(verts, codes)
        #pixelx = np.abs(np.diff(ax.get_window_extent().get_points()[:,0]))[0]
        #pixely = np.abs(np.diff(ax.get_window_extent().get_points()[:,1]))[0]
        #xmin, xmax = ax.get_xlim()
        #ymin, ymax = ax.get_ylim()
        #dxu = np.abs(xmax - xmin)/pixelx
        #dyu = np.abs(ymax - ymin)/pixely
        trans = mpl.transforms.Affine2D()
        angle = np.arctan2(direction[1], direction[0])
        trans.rotate(angle)
        #trans.scale(dxu/dyu, dyu/dxu)   # what is the right scaling????
        trans.scale(1, scaley)
        trans.translate(*pos)
        path = path.transformed(trans)
        kwargs = bodykwargs if part == 'body' else finkwargs
        ax.add_patch(PathPatch(path, **kwargs))


def plot_object(ax, pos=(0, 0), radius=1.0, **kwargs):
    """ Plot circular object.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the fish.
    pos: tuple of floats
        Coordinates of the objects's position (its center).
    radius: float
        Radius of the cirular object.
    kwargs: key word arguments
        Arguments for PathPatch used to draw the fish.
    """
    ax.add_patch(Circle(pos, radius, **kwargs))

    
def main():
    fig, ax = plt.subplots()
    fish1 = (('Alepto', 'top'), (-10, -7), (1, 1), 18.0, -25)
    fish2 = (('Alepto', 'side'), (10, 5), (1, -0.7), 20.0, 20)
    bodykwargs=dict(lw=1, edgecolor='k', facecolor='r')
    finkwargs=dict(lw=1, edgecolor='k', facecolor='b')
    plot_fish(ax, *fish1, bodykwargs=bodykwargs, finkwargs=finkwargs)
    plot_fish(ax, *fish2, bodykwargs=bodykwargs, finkwargs=finkwargs)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-20, 20)
    plt.show()


if __name__ == '__main__':
    #export_fish_demo()
    main()
