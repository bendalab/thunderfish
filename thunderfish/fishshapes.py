"""
Plot fish silhouettes.

- `plot_fish()`: plot silhouette of an electric fish.

For importing fish silhouettes use the following functions:
- `extract_fish()`: convert XML coordinates to numpy array.
- `mirror_fish()`: complete path of half a fish outline by appending the mirrored path.
- `normalize_fish()`: normalize fish outline to unit length.
- `export_fish()`: print coordinates of fish outline for import.
- `export_fish_demo(): code demonstrating how to export a fish outline.
"""

from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def extract_fish(data):
    """ Convert XML coordinates to numpy array.

    Draw a fish outline in inkscape. Open the XML Editor (shift+ctrl+x)
    and copy the value of the data field ('d') into a variable that you
    pas to this function.

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
    # from inkscape XML editor on Fish-Krahe2004-modified:
    coords = data.split(' ')
    xys = [c.split(',') for c in coords if ',' in c]
    rel = np.array([(float(x), float(y)) for x, y in xys])
    rel[0,:] = 0.0, 0.0
    vertices = np.cumsum(rel, axis=0)#[1:,:]
    return vertices


def mirror_fish(vertices1):
    """ Complete path of half a fish outline by appending the mirrored path.

    It is sufficient to draw half of a top view of a fish. Import with
    extract_fish() and use this function to add the missing half of the
    outline to the path.

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
    vertices2[:,0] *= -1
    vertices = np.concatenate((vertices1, vertices2))
    return vertices


def normalize_fish(vertices, midpoint=0.5):
    """ Normalize fish outline to unit length.

    Parameters
    ----------
    vertices: 2D array
        The coordinates of the outline of a fish.
    midpoint: float
        A number between 0 and 1 to which the center of the fish is shifted.

    Returns
    -------
    vertices: 2D array
        The normalized coordinates of the outline of a fish.
    """
    vertices[:,1] = -vertices[:,1]
    length = np.max(vertices[:,1]) - np.min(vertices[:,1])
    vertices[:,1] -= np.min(vertices[:,1])
    vertices /= length
    vertices[:,1] -= midpoint


def export_fish(vertices, n=2):
    """ Print coordinates of fish outline for import.

    Copy these coordinates from the console and paste them into this module.
    Give it a proper name and don't forget to add it to the fish_shapes dictionary
    to make it know to plot_fish().

    Parameters
    ----------
    vertices: 2D array
        The coordinates of the outline of a fish.
    n: int
        Number of coordinates printed in a single line.
    """
    print('fish_shape = np.array([')
    for k, v in enumerate(vertices):
        if k%n == 0:
            print('   ', end='')
        print(' [%.8e, %.8e],' % (v[0], v[1]), end='')
        if k%n == n-1 and k < len(vertices)-1:
            print('')
    print('])')


def export_fish_demo():
    """ Code demonstrating how to export a fish outline.
    """
    data = "m 84.013672,21.597656 0.0082,83.002434 c 0.07896,-0.0343 0.02497,0.0125 0.113201,-0.0145 0.07048,-0.0777 0.09577,-0.10334 0.1238,-0.32544 0.02803,-0.22209 0.0588,-0.64065 0.06532,-0.80506 l 0.06836,-0.87696 c 0.0027,-1.43338 -0.0229,-2.866455 0.0332,-4.298823 l 0,-8.625 c 0.0222,-0.574893 0.04531,-1.14975 0.06836,-1.724609 l 0.06836,-1.722657 c 0.02438,-0.575483 0.0473,-1.151024 0.07032,-1.726562 0.01938,-0.575658 0.04406,-1.151099 0.06836,-1.726563 0.02517,-0.564336 0.04565,-1.12885 0.06641,-1.693359 l 0.03439,-1.293583 0.06912,-1.30798 c 0.01085,-0.576459 0.05952,-1.150256 0.10547,-1.724609 0.0093,-0.576421 0.05679,-1.150236 0.10156,-1.724609 0.06003,-0.574112 0.08041,-1.150198 0.10352,-1.726563 0.06013,-0.573452 0.08044,-1.148892 0.10352,-1.724609 0.04961,-0.574599 0.09395,-1.149617 0.13867,-1.72461 0.08176,-0.856223 0.135607,-1.713266 0.171876,-2.572265 0.04751,-0.574717 0.092,-1.149671 0.13672,-1.72461 0.04739,-0.575377 0.09195,-1.150977 0.13672,-1.726562 0.06032,-0.57344 0.08049,-1.148887 0.10352,-1.724609 0.02515,-0.574126 0.04662,-1.148395 0.06836,-1.722657 l 0.103515,-2.574219 c 0.02036,-0.574317 0.04443,-1.148481 0.06836,-1.722656 0.06013,-0.574758 0.08044,-1.151498 0.10352,-1.728515 0.02438,-0.574181 0.0473,-1.148421 0.07032,-1.722657 0.02231,-0.574889 0.04536,-1.149748 0.06836,-1.724609 0.05585,-0.578351 0.149344,-1.1511 0.240234,-1.724609 0.164682,-0.56553 0.251806,-1.144455 0.34375,-1.72461 0.09933,-0.571989 0.114249,-1.148243 0.134766,-1.726562 0.01324,-0.565847 0.05933,-1.129383 0.10352,-1.69336 l 0.03516,-0.875 0.07031,-1.728515 0,-0.847657 -0.05312,-1.540271 -0.0172,-0.184338 0.15636,0.09441 1.090248,0.588297 1.153106,0.485031 1.330848,0.39517 0.738188,0.161698 -0.437272,-0.467071 -1.173608,-1.185447 -1.736597,-1.275371 -0.927443,-0.451153 -0.228986,-0.07018 -0.0015,-0.21624 0.03663,-0.660713 0.480469,-0.847657 -0.101563,-0.876953 -0.103515,-0.845703 -0.103516,-0.876953 c -0.06585,-0.565492 -0.136578,-1.130382 -0.207031,-1.695313 -0.08465,-0.575909 -0.179252,-1.15021 -0.273438,-1.724609 -0.07699,-0.58031 -0.193622,-1.153022 -0.308594,-1.726562 -0.08942,-0.57525 -0.182483,-1.149916 -0.27539,-1.72461 -0.07579,-0.579411 -0.194011,-1.15057 -0.310547,-1.722656 l -0.240234,-0.878906 -0.414063,-0.84961 -0.580871,-0.596268 -0.431105,-0.202816 z"
    verts = extract_fish(data)
    # fix the path:
    verts = verts[1:,:]
    verts[:,0] *= 0.8
    # mirrow, normalize and export path:
    verts = mirror_fish(verts)
    normalize_fish(verts, 0.45)
    export_fish(verts, 2)
    # plot outline:
    fig, ax = plt.subplots()
    plot_fish(ax, verts, size=-2.0*np.min(verts[:,1])/1.05,
              lw=1, edgecolor='k', facecolor='r')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1)
    plt.show()


""" Outline of an Apteronotus viewd from top, modified from Krahe 2004. """
apteronotus_top = np.array([
    [4.83732810e-05, -4.50000000e-01], [5.14172582e-04, -4.49747073e-01],
    [6.61475122e-04, -4.49839247e-01], [1.32926827e-03, -4.49732325e-01],
    [1.74504252e-03, -4.49159367e-01], [2.31000704e-03, -4.48397340e-01],
    [3.04032560e-03, -4.45997553e-01], [3.20567964e-03, -4.44359867e-01],
    [3.55255146e-03, -4.39635730e-01], [3.93788593e-03, -4.33699237e-01],
    [4.34115392e-03, -4.27232555e-01], [4.35708171e-03, -4.16662845e-01],
    [4.22199047e-03, -3.95525676e-01], [4.41784327e-03, -3.63826259e-01],
    [4.41784327e-03, -3.00225718e-01], [4.54880508e-03, -2.95986471e-01],
    [4.81609695e-03, -2.87508242e-01], [5.21936494e-03, -2.74791017e-01],
    [5.62263293e-03, -2.62088186e-01], [5.76645495e-03, -2.57844589e-01],
    [6.04548619e-03, -2.49356966e-01], [6.46031657e-03, -2.36625339e-01],
    [6.57464269e-03, -2.32380451e-01], [6.83456059e-03, -2.23892275e-01],
    [7.23782858e-03, -2.11160641e-01], [7.38631095e-03, -2.06999241e-01],
    [7.65560855e-03, -1.98675129e-01], [8.04737313e-03, -1.86188341e-01],
    [8.25024596e-03, -1.76649491e-01], [8.65799732e-03, -1.67004478e-01],
    [8.72200343e-03, -1.62753683e-01], [9.07312266e-03, -1.54271724e-01],
    [9.69530924e-03, -1.41554499e-01], [9.75017162e-03, -1.37303984e-01],
    [1.00851861e-02, -1.28822172e-01], [1.06843069e-02, -1.16104947e-01],
    [1.10384347e-02, -1.11871459e-01], [1.15127878e-02, -1.03389927e-01],
    [1.21234710e-02, -9.06582928e-02], [1.24781887e-02, -8.64296717e-02],
    [1.29527188e-02, -7.79577699e-02], [1.35634020e-02, -6.52405449e-02],
    [1.38560603e-02, -6.10034658e-02], [1.44102883e-02, -5.25262179e-02],
    [1.52283277e-02, -3.98089856e-02], [1.57106447e-02, -3.34952179e-02],
    [1.65106149e-02, -2.08616359e-02], [1.75245425e-02, -1.89381616e-03],
    [1.78048125e-02, 2.34413307e-03], [1.83475372e-02, 1.08217792e-02],
    [1.91540731e-02, 2.35390115e-02], [1.94336353e-02, 2.77818276e-02],
    [1.99760650e-02, 3.62691041e-02], [2.07826009e-02, 4.90007304e-02],
    [2.11384395e-02, 5.32292631e-02], [2.16132646e-02, 6.17011280e-02],
    [2.22239477e-02, 7.44183530e-02], [2.23723121e-02, 7.86519442e-02],
    [2.26473319e-02, 8.71201811e-02], [2.30505999e-02, 9.98230120e-02],
    [2.36612536e-02, 1.18805241e-01], [2.37813609e-02, 1.23040240e-01],
    [2.40434615e-02, 1.31509111e-01], [2.44467295e-02, 1.44211935e-01],
    [2.48014472e-02, 1.48450186e-01], [2.52759773e-02, 1.56941305e-01],
    [2.58866605e-02, 1.69687333e-01], [2.60304825e-02, 1.73921329e-01],
    [2.63095138e-02, 1.82389758e-01], [2.67243441e-02, 1.95092589e-01],
    [2.68559549e-02, 1.99331806e-01], [2.71235417e-02, 2.07810020e-01],
    [2.75268097e-02, 2.20527245e-01], [2.78562789e-02, 2.24791992e-01],
    [2.87372861e-02, 2.33280175e-01], [3.01544699e-02, 2.45997400e-01],
    [3.11259588e-02, 2.50167605e-01], [3.26114078e-02, 2.58606788e-01],
    [3.46392512e-02, 2.71324020e-01], [3.52252168e-02, 2.75541853e-01],
    [3.58991923e-02, 2.84008969e-01], [3.66942012e-02, 2.96740596e-01],
    [3.67723064e-02, 3.00913138e-01], [3.71223048e-02, 3.09241181e-01],
    [3.77329880e-02, 3.21727976e-01], [3.79404031e-02, 3.28180205e-01],
    [3.83551745e-02, 3.40926233e-01], [3.83551745e-02, 3.47176835e-01],
    [3.80418101e-02, 3.58534756e-01], [3.79403442e-02, 3.59894060e-01],
    [3.88627400e-02, 3.59197883e-01], [4.52943099e-02, 3.54859795e-01],
    [5.20966905e-02, 3.51283188e-01], [5.99476032e-02, 3.48369214e-01],
    [6.43023075e-02, 3.47176857e-01], [6.17227610e-02, 3.50621027e-01],
    [5.47994355e-02, 3.59362485e-01], [4.45549361e-02, 3.68767040e-01],
    [3.90837823e-02, 3.72093831e-01], [3.77329526e-02, 3.72611337e-01],
    [3.77241038e-02, 3.74205885e-01], [3.79401908e-02, 3.79077967e-01],
    [4.07745642e-02, 3.85328569e-01], [4.01754257e-02, 3.91795199e-01],
    [3.95647720e-02, 3.98031393e-01], [3.89541124e-02, 4.04498023e-01],
    [3.85656514e-02, 4.08667947e-01], [3.77599531e-02, 4.17003357e-01],
    [3.65386398e-02, 4.29504553e-01], [3.60392742e-02, 4.33751292e-01],
    [3.49818343e-02, 4.42232913e-01], [3.33687741e-02, 4.54950138e-01],
    [3.29145962e-02, 4.59229330e-01], [3.17723851e-02, 4.67731686e-01],
    [2.99519333e-02, 4.80463313e-01], [2.94244286e-02, 4.84705192e-01],
    [2.83479284e-02, 4.93184645e-01], [2.67233531e-02, 5.05901877e-01],
    [2.62762542e-02, 5.10174440e-01], [2.51317482e-02, 5.18658715e-01],
    [2.32997754e-02, 5.31361539e-01], [2.18825916e-02, 5.37842570e-01],
    [1.94399592e-02, 5.44107574e-01], [1.60132963e-02, 5.48504440e-01],
    [1.34701300e-02, 5.50000000e-01], [-1.34701300e-02, 5.50000000e-01],
    [-1.60132963e-02, 5.48504440e-01], [-1.94399592e-02, 5.44107574e-01],
    [-2.18825916e-02, 5.37842570e-01], [-2.32997754e-02, 5.31361539e-01],
    [-2.51317482e-02, 5.18658715e-01], [-2.62762542e-02, 5.10174440e-01],
    [-2.67233531e-02, 5.05901877e-01], [-2.83479284e-02, 4.93184645e-01],
    [-2.94244286e-02, 4.84705192e-01], [-2.99519333e-02, 4.80463313e-01],
    [-3.17723851e-02, 4.67731686e-01], [-3.29145962e-02, 4.59229330e-01],
    [-3.33687741e-02, 4.54950138e-01], [-3.49818343e-02, 4.42232913e-01],
    [-3.60392742e-02, 4.33751292e-01], [-3.65386398e-02, 4.29504553e-01],
    [-3.77599531e-02, 4.17003357e-01], [-3.85656514e-02, 4.08667947e-01],
    [-3.89541124e-02, 4.04498023e-01], [-3.95647720e-02, 3.98031393e-01],
    [-4.01754257e-02, 3.91795199e-01], [-4.07745642e-02, 3.85328569e-01],
    [-3.79401908e-02, 3.79077967e-01], [-3.77241038e-02, 3.74205885e-01],
    [-3.77329526e-02, 3.72611337e-01], [-3.90837823e-02, 3.72093831e-01],
    [-4.45549361e-02, 3.68767040e-01], [-5.47994355e-02, 3.59362485e-01],
    [-6.17227610e-02, 3.50621027e-01], [-6.43023075e-02, 3.47176857e-01],
    [-5.99476032e-02, 3.48369214e-01], [-5.20966905e-02, 3.51283188e-01],
    [-4.52943099e-02, 3.54859795e-01], [-3.88627400e-02, 3.59197883e-01],
    [-3.79403442e-02, 3.59894060e-01], [-3.80418101e-02, 3.58534756e-01],
    [-3.83551745e-02, 3.47176835e-01], [-3.83551745e-02, 3.40926233e-01],
    [-3.79404031e-02, 3.28180205e-01], [-3.77329880e-02, 3.21727976e-01],
    [-3.71223048e-02, 3.09241181e-01], [-3.67723064e-02, 3.00913138e-01],
    [-3.66942012e-02, 2.96740596e-01], [-3.58991923e-02, 2.84008969e-01],
    [-3.52252168e-02, 2.75541853e-01], [-3.46392512e-02, 2.71324020e-01],
    [-3.26114078e-02, 2.58606788e-01], [-3.11259588e-02, 2.50167605e-01],
    [-3.01544699e-02, 2.45997400e-01], [-2.87372861e-02, 2.33280175e-01],
    [-2.78562789e-02, 2.24791992e-01], [-2.75268097e-02, 2.20527245e-01],
    [-2.71235417e-02, 2.07810020e-01], [-2.68559549e-02, 1.99331806e-01],
    [-2.67243441e-02, 1.95092589e-01], [-2.63095138e-02, 1.82389758e-01],
    [-2.60304825e-02, 1.73921329e-01], [-2.58866605e-02, 1.69687333e-01],
    [-2.52759773e-02, 1.56941305e-01], [-2.48014472e-02, 1.48450186e-01],
    [-2.44467295e-02, 1.44211935e-01], [-2.40434615e-02, 1.31509111e-01],
    [-2.37813609e-02, 1.23040240e-01], [-2.36612536e-02, 1.18805241e-01],
    [-2.30505999e-02, 9.98230120e-02], [-2.26473319e-02, 8.71201811e-02],
    [-2.23723121e-02, 7.86519442e-02], [-2.22239477e-02, 7.44183530e-02],
    [-2.16132646e-02, 6.17011280e-02], [-2.11384395e-02, 5.32292631e-02],
    [-2.07826009e-02, 4.90007304e-02], [-1.99760650e-02, 3.62691041e-02],
    [-1.94336353e-02, 2.77818276e-02], [-1.91540731e-02, 2.35390115e-02],
    [-1.83475372e-02, 1.08217792e-02], [-1.78048125e-02, 2.34413307e-03],
    [-1.75245425e-02, -1.89381616e-03], [-1.65106149e-02, -2.08616359e-02],
    [-1.57106447e-02, -3.34952179e-02], [-1.52283277e-02, -3.98089856e-02],
    [-1.44102883e-02, -5.25262179e-02], [-1.38560603e-02, -6.10034658e-02],
    [-1.35634020e-02, -6.52405449e-02], [-1.29527188e-02, -7.79577699e-02],
    [-1.24781887e-02, -8.64296717e-02], [-1.21234710e-02, -9.06582928e-02],
    [-1.15127878e-02, -1.03389927e-01], [-1.10384347e-02, -1.11871459e-01],
    [-1.06843069e-02, -1.16104947e-01], [-1.00851861e-02, -1.28822172e-01],
    [-9.75017162e-03, -1.37303984e-01], [-9.69530924e-03, -1.41554499e-01],
    [-9.07312266e-03, -1.54271724e-01], [-8.72200343e-03, -1.62753683e-01],
    [-8.65799732e-03, -1.67004478e-01], [-8.25024596e-03, -1.76649491e-01],
    [-8.04737313e-03, -1.86188341e-01], [-7.65560855e-03, -1.98675129e-01],
    [-7.38631095e-03, -2.06999241e-01], [-7.23782858e-03, -2.11160641e-01],
    [-6.83456059e-03, -2.23892275e-01], [-6.57464269e-03, -2.32380451e-01],
    [-6.46031657e-03, -2.36625339e-01], [-6.04548619e-03, -2.49356966e-01],
    [-5.76645495e-03, -2.57844589e-01], [-5.62263293e-03, -2.62088186e-01],
    [-5.21936494e-03, -2.74791017e-01], [-4.81609695e-03, -2.87508242e-01],
    [-4.54880508e-03, -2.95986471e-01], [-4.41784327e-03, -3.00225718e-01],
    [-4.41784327e-03, -3.63826259e-01], [-4.22199047e-03, -3.95525676e-01],
    [-4.35708171e-03, -4.16662845e-01], [-4.34115392e-03, -4.27232555e-01],
    [-3.93788593e-03, -4.33699237e-01], [-3.55255146e-03, -4.39635730e-01],
    [-3.20567964e-03, -4.44359867e-01], [-3.04032560e-03, -4.45997553e-01],
    [-2.31000704e-03, -4.48397340e-01], [-1.74504252e-03, -4.49159367e-01],
    [-1.32926827e-03, -4.49732325e-01], [-6.61475122e-04, -4.49839247e-01],
    [-5.14172582e-04, -4.49747073e-01], [-4.83732810e-05, -4.50000000e-01]])


""" Dictionary holding all known electric fish shapes. """
fish_shapes = dict(apterotop=apteronotus_top)
    

def plot_fish(ax, fish, pos=(0, 0), direction=(0, 1), size=20.0, bend=0, scaley=1, **kwargs):
    """ Plot silhouette of an electric fish.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the fish.
    fish: string or 2D array
        Specifies which fish to show. Either any of the strings defining a shape
        contained in the `fish_shapes` dictionary, or a list of vertices.
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
    kwargs: key word arguments
        Arguments for PathPatch used to draw the fish.
    """
    if isinstance(fish, np.ndarray):
        verts = np.array(fish)
    else:
        verts = np.array(fish_shapes[fish])
    size_fac = -1.05*0.5/np.min(verts[:,1])
    verts *= size_fac*size
    if np.abs(bend) > 1.e-8:
        sel = verts[:,1]<0.0
        xp = verts[sel,0]   # x coordinates of all negative y coordinates of verts
        yp = verts[sel,1]   # all negative y coordinates of verts
        r = 180.0*0.5*size/bend/np.pi               # radius of circle on which to bend the tail
        beta = yp/r                                 # angle on circle for each y coordinate
        verts[sel,0] = r-(r+xp)*np.cos(beta)        # transformed x corrdinates
        verts[sel,1] = -np.abs((r+xp)*np.sin(beta)) # transformed y coordinates
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
    angle = np.arctan2(-direction[0], direction[1])
    trans.rotate(angle)
    #trans.scale(dxu/dyu, dyu/dxu)   # what is the right scaling????
    trans.scale(1, scaley)
    trans.translate(*pos)
    path = path.transformed(trans)
    ax.add_patch(PathPatch(path, **kwargs))

    
def main():
    fig, ax = plt.subplots()
    fish1 = ('apterotop', (-10, -7), (1, 0.5), 18.0, -25)
    fish2 = ('apterotop', (10, 1), (0.8, 1), 20.0, 10)
    plot_fish(ax, *fish1, lw=1, edgecolor='k', facecolor='k')
    plot_fish(ax, *fish2, lw=1, edgecolor='k', facecolor='k')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-20, 20)
    plt.show()


if __name__ == '__main__':
    #export_fish_demo()
    main()
