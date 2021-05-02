"""
Manipulate and plot fish outlines.

## Fish shapes

All fish shapes of this module are accessible via these dictionaries:

- `fish_shapes`: dictionary holding all electric fish shapes.
- `fish_top_shapes`: dictionary holding electric fish shapes viewed from top.
- `fish_side_shapes`: dictionary holding electric fish shapes viewed from the side.

These are the shapes of various fish species:

- `Alepto_top`: *Apteronotus leptorhynchus* viewed from top.
- `Alepto_male_side`: Male *Apteronotus leptorhynchus* viewed from the side.
- `Eigenmannia_top`: *Eigenmannia virescens* viewed from top.
- `Eigenmannia_side`: *Eigenmannia virescens* viewed from the side.

## Plotting

- `plot_fish()`: plot body and fin of an electric fish.
- `plot_object()`: plot circular object.
- `plot_pathes()`: plot pathes.

## Fish surface and normals from shapes

- `fish_surface()`: generate meshgrid of one side of the fish from shape.
- `surface_normals()`: normal vectors on a surface.

## General path manipulations

You may use these functions to extract and fine tune pathes from SVG files in order
to assemble fish shapes for this module. See `export_fish_demo()` for a use case.

- `extract_path()`: convert SVG coordinates to numpy array with path coordinates.
- `bbox_pathes()`: common bounding box of pathes.
- `translate_pathes()`: translate pathes in place.
- `center_pathes()`: translate pathes to their common origin in place.
- `rotate_pathes()`: rotate pathes in place.
- `flipy_pathes()`: flip pathes in y-direction in place.
- `flipx_pathes()`: flip pathes in x-direction in place.
- `export_path()`: print coordinates of path for import as numpy array.
- `mirror_path()`: complete path of half a fish outline by appending the mirrored path.
- `normalize_path()`: normalize fish outline to unit length.
- `bend_path()`: bend and scale a path.

## Exporting fish outlines from pathes

- `export_fish()`: serialize coordinates of fish outlines as a dictionary.
- `export_fish_demo()`: code demonstrating how to export fish outlines from SVG.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


Alepto_top = dict(body=np.array([
    [-5.00000000e-01, 0.00000000e+00], [-4.99802704e-01, 1.23222860e-03],
    [-4.95374557e-01, 2.57983066e-03], [-4.84420392e-01, 3.29085947e-03],
    [-4.72487909e-01, 4.03497963e-03], [-4.13995354e-01, 4.39637211e-03],
    [-3.90529212e-01, 5.14049228e-03], [-3.67089631e-01, 5.88461244e-03],
    [-3.43596916e-01, 6.65006783e-03], [-3.20104187e-01, 7.39418800e-03],
    [-2.97063253e-01, 8.11708180e-03], [-2.79461930e-01, 8.49142780e-03],
    [-2.61664711e-01, 9.24382081e-03], [-2.38198570e-01, 1.03918950e-02],
    [-2.14732428e-01, 1.14974077e-02], [-1.91239699e-01, 1.26242555e-02],
    [-1.67773558e-01, 1.37511034e-02], [-1.44307403e-01, 1.52605701e-02],
    [-1.09307508e-01, 1.71314946e-02], [-8.58413531e-02, 1.86197349e-02],
    [-6.23486380e-02, 2.01079753e-02], [-3.88824966e-02, 2.12348231e-02],
    [-1.54429155e-02, 2.19789433e-02], [1.95835670e-02, 2.31057367e-02],
    [4.30231346e-02, 2.38498569e-02], [6.65424230e-02, 2.49767047e-02],
    [8.99820050e-02, 2.57421601e-02], [1.13448146e-01, 2.64862803e-02],
    [1.36914287e-01, 2.91013032e-02], [1.60380442e-01, 3.28431304e-02],
    [1.83873157e-01, 3.43101008e-02], [2.06914105e-01, 3.54369487e-02],
    [2.18819919e-01, 3.58196764e-02], [2.42339207e-01, 3.65850229e-02],
    [2.72903364e-01, 3.57933339e-02], [2.75411585e-01, 3.56061065e-02],
    [2.74126982e-01, 3.73081344e-02], [2.60251756e-01, 4.14908387e-02],
    [2.47930096e-01, 4.96419915e-02], [2.39119358e-01, 6.08413919e-02],
    [2.39547832e-01, 7.56059835e-02], [2.44279733e-01, 7.95534778e-02],
    [2.54298155e-01, 7.66782524e-02], [2.69627591e-01, 6.28724285e-02],
    [2.82177993e-01, 4.80249888e-02], [2.88316671e-01, 3.79294791e-02],
    [2.89271585e-01, 3.54368942e-02], [2.92213886e-01, 3.54205663e-02],
    [3.01203973e-01, 3.58192954e-02], [3.12737740e-01, 4.10493520e-02],
    [3.24670128e-01, 3.99438067e-02], [3.36177308e-01, 3.88170133e-02],
    [3.48109696e-01, 3.76902090e-02], [3.71177217e-01, 3.54366112e-02],
    [3.94643358e-01, 3.24601523e-02], [4.18136073e-01, 2.91010093e-02],
    [4.41602228e-01, 2.61033022e-02], [4.65041796e-01, 2.27229002e-02],
    [4.77000757e-01, 2.01078773e-02], [4.88938465e-01, 1.57516176e-02],
    [4.97051671e-01, 9.88149348e-03], [5.00000000e-01, 4.58499286e-03],
    [5.00000000e-01, -4.58499286e-03], [4.97051671e-01, -9.88149348e-03],
    [4.88938465e-01, -1.57516176e-02], [4.77000757e-01, -2.01078773e-02],
    [4.65041796e-01, -2.27229002e-02], [4.41602228e-01, -2.61033022e-02],
    [4.18136073e-01, -2.91010093e-02], [3.94643358e-01, -3.24601523e-02],
    [3.71177217e-01, -3.54366112e-02], [3.48109696e-01, -3.76902090e-02],
    [3.36177308e-01, -3.88170133e-02], [3.24670128e-01, -3.99438067e-02],
    [3.12737740e-01, -4.10493520e-02], [3.01203973e-01, -3.58192954e-02],
    [2.92213886e-01, -3.54205663e-02], [2.89271585e-01, -3.54368942e-02],
    [2.88316671e-01, -3.79294791e-02], [2.82177993e-01, -4.80249888e-02],
    [2.69627591e-01, -6.28724285e-02], [2.54298155e-01, -7.66782524e-02],
    [2.44279733e-01, -7.95534778e-02], [2.39547832e-01, -7.56059835e-02],
    [2.39119358e-01, -6.08413919e-02], [2.47930096e-01, -4.96419915e-02],
    [2.60251756e-01, -4.14908387e-02], [2.74126982e-01, -3.73081344e-02],
    [2.75411585e-01, -3.56061065e-02], [2.72903364e-01, -3.57933339e-02],
    [2.42339207e-01, -3.65850229e-02], [2.18819919e-01, -3.58196764e-02],
    [2.06914105e-01, -3.54369487e-02], [1.83873157e-01, -3.43101008e-02],
    [1.60380442e-01, -3.28431304e-02], [1.36914287e-01, -2.91013032e-02],
    [1.13448146e-01, -2.64862803e-02], [8.99820050e-02, -2.57421601e-02],
    [6.65424230e-02, -2.49767047e-02], [4.30231346e-02, -2.38498569e-02],
    [1.95835670e-02, -2.31057367e-02], [-1.54429155e-02, -2.19789433e-02],
    [-3.88824966e-02, -2.12348231e-02], [-6.23486380e-02, -2.01079753e-02],
    [-8.58413531e-02, -1.86197349e-02], [-1.09307508e-01, -1.71314946e-02],
    [-1.44307403e-01, -1.52605701e-02], [-1.67773558e-01, -1.37511034e-02],
    [-1.91239699e-01, -1.26242555e-02], [-2.14732428e-01, -1.14974077e-02],
    [-2.38198570e-01, -1.03918950e-02], [-2.61664711e-01, -9.24382081e-03],
    [-2.79461930e-01, -8.49142780e-03], [-2.97063253e-01, -8.11708180e-03],
    [-3.20104187e-01, -7.39418800e-03], [-3.43596916e-01, -6.65006783e-03],
    [-3.67089631e-01, -5.88461244e-03], [-3.90529212e-01, -5.14049228e-03],
    [-4.13995354e-01, -4.39637211e-03], [-4.72487909e-01, -4.03497963e-03],
    [-4.84420392e-01, -3.29085947e-03], [-4.95374557e-01, -2.57983066e-03],
    [-4.99802704e-01, -1.23222860e-03], [-5.00000000e-01, -0.00000000e+00],]))
""" Outline of an *Apteronotus leptorhynchus* viewed from top, modified from Krahe 2004. """

Alepto_male_side = dict(body=np.array([
    [2.80332097e-01, 5.51361973e-02], [2.41127905e-01, 5.93460338e-02],
    [1.91463866e-01, 6.22667811e-02], [1.37379023e-01, 6.17716006e-02],
    [6.91234340e-02, 5.72953633e-02], [-1.36051588e-02, 4.74838393e-02],
    [-7.55221954e-02, 3.64211032e-02], [-1.60157310e-01, 2.45651115e-02],
    [-2.32035003e-01, 1.55421483e-02], [-2.99079447e-01, 9.70960800e-03],
    [-3.62251791e-01, 6.27265707e-03], [-4.20527920e-01, 4.22449025e-03],
    [-4.72735573e-01, 5.39606712e-03], [-4.80154179e-01, 5.86398206e-03],
    [-4.92605065e-01, 1.01411700e-02], [-4.97402289e-01, 5.91543079e-03],
    [-5.00000000e-01, -2.84973497e-03], [-4.97832769e-01, -1.17981289e-02],
    [-4.93106950e-01, -1.43380199e-02], [-4.81164618e-01, -8.19215843e-03],
    [-4.72578673e-01, -6.17623988e-03], [-4.45390092e-01, -5.96123217e-03],
    [-3.74805165e-01, -9.05994885e-03], [-3.33716813e-01, -1.08317142e-02],
    [-3.08099380e-01, -1.15017063e-02], [-2.82451613e-01, -1.30396176e-02],
    [-2.34498580e-01, -2.21834040e-02], [-1.86892658e-01, -3.26728000e-02],
    [-1.08738732e-01, -4.99024273e-02], [-3.50753879e-02, -5.94218882e-02],
    [3.28767168e-02, -6.58397526e-02], [1.25319086e-01, -7.21513968e-02],
    [1.99523049e-01, -7.99740378e-02], [2.37035792e-01, -8.44828747e-02],
    [2.74475366e-01, -8.68964223e-02], [3.12742824e-01, -8.34038539e-02],
    [3.36340505e-01, -7.82231053e-02], [3.55492327e-01, -7.21451373e-02],
    [3.74670470e-01, -6.45564453e-02], [3.82920881e-01, -6.06824741e-02],
    [3.84828678e-01, -5.92550189e-02], [3.86562866e-01, -5.99353293e-02],
    [3.90753372e-01, -6.01589140e-02], [4.03494946e-01, -5.90960625e-02],
    [4.38474761e-01, -6.13270959e-02], [4.61389913e-01, -6.47960654e-02],
    [4.77010163e-01, -6.86433853e-02], [4.84437594e-01, -6.89404377e-02],
    [4.90842798e-01, -6.82840746e-02], [4.94567181e-01, -6.58050993e-02],
    [4.95443985e-01, -6.30972916e-02], [4.94497789e-01, -6.10849673e-02],
    [4.91729699e-01, -6.00016418e-02], [4.84298546e-01, -5.78808424e-02],
    [4.93112897e-01, -5.45550751e-02], [4.97742360e-01, -5.12667865e-02],
    [5.00000000e-01, -4.73196051e-02], [4.99521047e-01, -4.36153642e-02],
    [4.96159278e-01, -3.87756472e-02], [4.86402575e-01, -3.18513601e-02],
    [4.67134496e-01, -2.06920393e-02], [4.39218141e-01, -5.92866768e-03],
    [4.25010402e-01, 4.45359743e-03], [4.14788070e-01, 1.39860522e-02],
    [3.93656086e-01, 2.44160739e-02], [3.75679976e-01, 2.94323719e-02],
    [3.61404254e-01, 3.69002336e-02], [3.37900061e-01, 4.40458301e-02],
    [3.11463577e-01, 4.97553861e-02],]),
    fin0=np.array([
    [3.29593304e-01, -7.95912942e-02], [3.27561074e-01, -8.48367727e-02],
    [3.08709726e-01, -9.90609655e-02], [2.80934315e-01, -1.08062137e-01],
    [2.58017473e-01, -1.12878542e-01], [2.35142157e-01, -1.14467112e-01],
    [2.18081531e-01, -1.12354592e-01], [1.98185626e-01, -1.10721292e-01],
    [1.78099090e-01, -1.13640193e-01], [1.59752865e-01, -1.18762090e-01],
    [1.40752841e-01, -1.20266781e-01], [1.27904629e-01, -1.17712356e-01],
    [1.19134213e-01, -1.12284346e-01], [1.09580014e-01, -1.04436264e-01],
    [8.20184710e-02, -9.60992771e-02], [5.05598670e-02, -9.57289587e-02],
    [2.74790284e-02, -1.04021601e-01], [3.92704920e-03, -1.08834461e-01],
    [-3.12710137e-02, -1.08965162e-01], [-5.88865488e-02, -1.03820945e-01],
    [-7.82549598e-02, -9.45428978e-02], [-9.94601687e-02, -8.20174601e-02],
    [-1.29941640e-01, -7.01658118e-02], [-1.58259295e-01, -6.73695625e-02],
    [-1.86001442e-01, -7.01570717e-02], [-2.14339679e-01, -6.79007296e-02],
    [-2.38708971e-01, -5.78982409e-02], [-2.55168178e-01, -4.41230328e-02],
    [-2.71293058e-01, -3.28785160e-02], [-2.88416341e-01, -2.86291802e-02],
    [-3.06103856e-01, -2.82461534e-02], [-3.22345146e-01, -2.47128040e-02],
    [-3.38333410e-01, -1.44124470e-02], [-3.43264223e-01, -1.03691894e-02],
    [-3.08609907e-01, -1.12571357e-02], [-2.86088545e-01, -1.25633719e-02],
    [-2.59977440e-01, -1.65414204e-02], [-2.16119429e-01, -2.64072955e-02],
    [-1.68443229e-01, -3.68996138e-02], [-1.12717944e-01, -4.88585839e-02],
    [-7.07908982e-02, -5.51259999e-02], [-1.80906639e-02, -6.16068166e-02],
    [2.75299392e-02, -6.53080983e-02], [7.71390030e-02, -6.85205021e-02],
    [1.21071140e-01, -7.25104674e-02], [1.78723549e-01, -7.85286909e-02],
    [2.32100395e-01, -8.40268652e-02], [2.74938812e-01, -8.74456073e-02],
    [3.10041908e-01, -8.43007220e-02],]),
    eye=np.array([0.4, 0.0, 0.01]))
""" Outline of an *Apteronotus leptorhynchus* male viewed from the side. """

Eigenmannia_top = dict(body=np.array([
    [-5.00000000e-01, 0.00000000e+00], [-4.84515329e-01, 4.41536208e-03],
    [-4.76913801e-01, 5.34924846e-03], [-3.94680346e-01, 8.25734868e-03],
    [-2.74106007e-01, 8.94059314e-03], [-1.35145770e-01, 1.09559947e-02],
    [2.36080412e-02, 1.40941342e-02], [1.36968804e-01, 1.51550643e-02],
    [2.15041020e-01, 1.96734219e-02], [2.83582110e-01, 2.36895289e-02],
    [3.20834553e-01, 2.63067663e-02], [3.46646908e-01, 2.77590937e-02],
    [3.68462758e-01, 2.97229886e-02], [3.62525174e-01, 3.12766064e-02],
    [3.57215426e-01, 3.25163153e-02], [3.51347983e-01, 3.44809486e-02],
    [3.46108357e-01, 3.83290703e-02], [3.44207747e-01, 4.53621620e-02],
    [3.46387987e-01, 5.39648157e-02], [3.54784122e-01, 6.69720204e-02],
    [3.67470562e-01, 8.11691502e-02], [3.80987875e-01, 9.13148567e-02],
    [3.90738756e-01, 9.39276818e-02], [3.95854520e-01, 9.06728175e-02],
    [3.99717109e-01, 8.49081236e-02], [3.96997843e-01, 6.54750599e-02],
    [3.89101023e-01, 4.11631100e-02], [3.86289062e-01, 3.71837960e-02],
    [3.94553267e-01, 3.78052325e-02], [4.03373690e-01, 3.72181278e-02],
    [4.20207675e-01, 3.56696607e-02], [4.37553246e-01, 3.46018748e-02],
    [4.59139056e-01, 3.15068918e-02], [4.79811600e-01, 2.68634593e-02],
    [4.92810472e-01, 1.97499259e-02], [4.98594784e-01, 1.11517021e-02],
    [5.00000000e-01, 5.62393850e-03], [5.00000000e-01, -5.62393850e-03],
    [4.98594784e-01, -1.11517021e-02], [4.92810472e-01, -1.97499259e-02],
    [4.79811600e-01, -2.68634593e-02], [4.59139056e-01, -3.15068918e-02],
    [4.37553246e-01, -3.46018748e-02], [4.20207675e-01, -3.56696607e-02],
    [4.03373690e-01, -3.72181278e-02], [3.94553267e-01, -3.78052325e-02],
    [3.86289062e-01, -3.71837960e-02], [3.89101023e-01, -4.11631100e-02],
    [3.96997843e-01, -6.54750599e-02], [3.99717109e-01, -8.49081236e-02],
    [3.95854520e-01, -9.06728175e-02], [3.90738756e-01, -9.39276818e-02],
    [3.80987875e-01, -9.13148567e-02], [3.67470562e-01, -8.11691502e-02],
    [3.54784122e-01, -6.69720204e-02], [3.46387987e-01, -5.39648157e-02],
    [3.44207747e-01, -4.53621620e-02], [3.46108357e-01, -3.83290703e-02],
    [3.51347983e-01, -3.44809486e-02], [3.57215426e-01, -3.25163153e-02],
    [3.62525174e-01, -3.12766064e-02], [3.68462758e-01, -2.97229886e-02],
    [3.46646908e-01, -2.77590937e-02], [3.20834553e-01, -2.63067663e-02],
    [2.83582110e-01, -2.36895289e-02], [2.15041020e-01, -1.96734219e-02],
    [1.36968804e-01, -1.51550643e-02], [2.36080412e-02, -1.40941342e-02],
    [-1.35145770e-01, -1.09559947e-02], [-2.74106007e-01, -8.94059314e-03],
    [-3.94680346e-01, -8.25734868e-03], [-4.76913801e-01, -5.34924846e-03],
    [-4.84515329e-01, -4.41536208e-03], [-5.00000000e-01, -0.00000000e+00],]))
""" Outline of an *Eigenmannia virescens* viewed from top. """

Eigenmannia_side = dict(body=np.array([
    [7.39835590e-02, 4.57421567e-02], [1.36190672e-01, 5.20008556e-02],
    [1.88575637e-01, 5.31087788e-02], [2.55693889e-01, 4.90162062e-02],
    [2.91989388e-01, 4.57421567e-02], [3.30997244e-01, 4.08310609e-02],
    [3.60079352e-01, 3.50312357e-02], [3.86267547e-01, 2.72057399e-02],
    [4.09748495e-01, 1.88510343e-02], [4.30914243e-01, 1.02069720e-02],
    [4.43253678e-01, 5.18028074e-03], [4.61959655e-01, -3.75313831e-03],
    [4.82422519e-01, -1.50677197e-02], [4.93493046e-01, -2.26243878e-02],
    [4.97325280e-01, -2.75603439e-02], [5.00000000e-01, -3.36538136e-02],
    [4.99855343e-01, -3.81556262e-02], [4.97829629e-01, -4.26574388e-02],
    [4.95229403e-01, -4.49683083e-02], [4.93207934e-01, -4.68450344e-02],
    [4.90607707e-01, -4.83870578e-02], [4.92124870e-01, -5.04085273e-02],
    [4.93063234e-01, -5.27193968e-02], [4.93063190e-01, -5.47905000e-02],
    [4.91905677e-01, -5.65722031e-02], [4.87982621e-01, -5.83539496e-02],
    [4.81889151e-01, -5.99909526e-02], [4.72187579e-01, -6.31614903e-02],
    [4.57251469e-01, -6.96684443e-02], [4.42315315e-01, -7.44390846e-02],
    [4.31434877e-01, -7.64563096e-02], [4.21852452e-01, -8.03091592e-02],
    [4.12030260e-01, -8.11773161e-02], [3.97297016e-01, -8.61380457e-02],
    [3.84200775e-01, -9.05200184e-02], [3.71589870e-01, -9.38291926e-02],
    [3.58008292e-01, -9.50424035e-02], [3.33452813e-01, -9.34053571e-02],
    [2.99075185e-01, -8.68572582e-02], [2.70427177e-01, -8.11276391e-02],
    [2.32775500e-01, -7.31023958e-02], [2.00034918e-01, -6.81912999e-02],
    [1.71386866e-01, -6.43481085e-02], [1.37488988e-01, -5.96768656e-02],
    [8.87168470e-02, -5.53444400e-02], [3.71504052e-02, -5.08426274e-02],
    [-8.94935470e-03, -4.47741911e-02], [-6.68009664e-02, -3.60218095e-02],
    [-1.11819296e-01, -3.02864735e-02], [-1.55609841e-01, -2.46444281e-02],
    [-2.01855938e-01, -1.98208625e-02], [-2.61607520e-01, -1.41655641e-02],
    [-3.02124011e-01, -9.83500080e-03], [-3.47551590e-01, -8.19795443e-03],
    [-3.86021794e-01, -7.21576125e-03], [-4.19580907e-01, -5.90618477e-03],
    [-4.49047446e-01, -5.00584824e-03], [-4.82606558e-01, -4.29793979e-03],
    [-4.93367213e-01, -3.88865654e-03], [-4.96609514e-01, -3.33497643e-03],
    [-4.98599358e-01, -2.28352992e-03], [-5.00000000e-01, -4.13646830e-04],
    [-4.99911798e-01, 1.42799787e-03], [-4.97749085e-01, 3.02268669e-03],
    [-4.94153971e-01, 3.94706050e-03], [-4.48842818e-01, 5.27946155e-03],
    [-3.90932887e-01, 5.88974836e-03], [-3.04988822e-01, 7.10408527e-03],
    [-2.43785835e-01, 8.93052803e-03], [-1.87718481e-01, 1.20250559e-02],
    [-1.39987578e-01, 1.55534240e-02], [-9.58582596e-02, 1.92113768e-02],
    [-4.87936436e-02, 2.54739303e-02], [-1.20172913e-02, 3.11685979e-02],
    [3.65545828e-02, 3.98634200e-02],]),
    fin0=np.array([
    [-3.23227396e-01, -8.73526322e-03], [-3.17729007e-01, -1.49720903e-02],
    [-3.11901320e-01, -2.06301173e-02], [-2.94537996e-01, -2.87329729e-02],
    [-2.73702014e-01, -3.62471102e-02], [-2.48814582e-01, -4.42901541e-02],
    [-2.26392044e-01, -4.89203820e-02], [-2.11413629e-01, -4.97652813e-02],
    [-1.97592770e-01, -4.71608105e-02], [-1.88292360e-01, -4.37113973e-02],
    [-1.77575020e-01, -4.26201918e-02], [-1.63230314e-01, -4.13425351e-02],
    [-1.45633053e-01, -4.58128611e-02], [-1.32102997e-01, -5.21132245e-02],
    [-1.22627830e-01, -5.98022925e-02], [-1.16274541e-01, -6.51393895e-02],
    [-1.01226326e-01, -6.99292162e-02], [-8.87826127e-02, -7.09420732e-02],
    [-7.63388990e-02, -7.02186163e-02], [-6.41845810e-02, -6.63566715e-02],
    [-4.99997329e-02, -6.35107453e-02], [-3.86044383e-02, -6.71556184e-02],
    [-2.83003535e-02, -7.56835222e-02], [-1.41203129e-02, -8.28817968e-02],
    [1.21728460e-03, -8.66205668e-02], [1.22140543e-02, -8.75385740e-02],
    [2.16240177e-02, -8.43285373e-02], [3.27836777e-02, -8.13081568e-02],
    [3.98554860e-02, -8.02952999e-02], [4.86770343e-02, -7.96350762e-02],
    [5.81904230e-02, -8.20450399e-02], [6.47198980e-02, -8.65937577e-02],
    [7.29857310e-02, -9.36024194e-02], [8.47509570e-02, -9.91141438e-02],
    [1.00477612e-01, -1.02776515e-01], [1.28258936e-01, -1.02826321e-01],
    [1.45605097e-01, -1.02460349e-01], [1.59342462e-01, -9.97657918e-02],
    [1.76140399e-01, -9.72111283e-02], [1.89366052e-01, -9.61800377e-02],
    [2.03938918e-01, -9.84587276e-02], [2.14786136e-01, -1.02170949e-01],
    [2.24046592e-01, -1.08953357e-01], [2.34464605e-01, -1.14112491e-01],
    [2.47925953e-01, -1.18114112e-01], [2.65013334e-01, -1.19108779e-01],
    [2.83520819e-01, -1.15835465e-01], [2.98329467e-01, -1.08650574e-01],
    [3.15014321e-01, -1.04499489e-01], [3.28805304e-01, -1.04273408e-01],
    [3.39387031e-01, -1.06211982e-01], [3.52278630e-01, -1.03431974e-01],
    [3.61896180e-01, -1.00567165e-01], [3.67032403e-01, -9.80662488e-02],
    [3.71589870e-01, -9.38289761e-02], [3.58008292e-01, -9.50421869e-02],
    [3.33452813e-01, -9.34051405e-02], [3.06441808e-01, -8.84940880e-02],
    [2.35043362e-01, -7.35981699e-02], [1.65011316e-01, -6.31802003e-02],
    [1.25654422e-01, -5.85499724e-02], [9.49792270e-02, -5.56561016e-02],
    [4.05741354e-02, -5.11056947e-02], [-1.24746680e-03, -4.58268936e-02],
    [-5.20302500e-02, -3.81131387e-02], [-1.01805114e-01, -3.16101258e-02],
    [-1.51874267e-01, -2.50855445e-02], [-2.01943420e-01, -2.02074944e-02],
    [-2.61607516e-01, -1.41653476e-02], [-3.02124016e-01, -9.83478430e-03],
    [-3.12840355e-01, -9.28491550e-03],]),
    eye=np.array([0.46, -0.03, 0.005]))
""" Outline of an *Eigenmannia virescens* viewed from the side. """

fish_shapes = dict(Alepto_top=Alepto_top,
                   Alepto_male_side=Alepto_male_side,
                   Eigenmannia_top=Eigenmannia_top,
                   Eigenmannia_side=Eigenmannia_side)
""" Dictionary holding all electric fish shapes. """

fish_top_shapes = dict(Alepto=Alepto_top,
                       Eigenmannia=Eigenmannia_top)
""" Dictionary holding electric fish shapes viewed from top. """

fish_side_shapes = dict(Alepto_male=Alepto_male_side,
                   Eigenmannia=Eigenmannia_side)
""" Dictionary holding electric fish shapes viewed from the side. """
    

def plot_fish(ax, fish, pos=(0, 0), direction=(1, 0), size=20.0, bend=0, scaley=1,
              bodykwargs={}, finkwargs={}, eyekwargs=None):
    """ Plot body and fin of an electric fish.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the fish.
    fish: string or tuple or dict
        Specifies a fish to show:
        - any of the strings defining a shape contained in the `fish_shapes` dictionary,
        - a tuple with the name of the fish as the first element and 'top' or 'side' as the second element,
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
        compensate for differently scaled axes.
    bodykwargs: dict
        Key-word arguments for PathPatch used to draw the fish's body.
    finkwargs: dict
        Key-word arguments for PathPatch used to draw the fish's fins.

    Returns
    -------
    bpatch: matplotlib.patches.PathPatch
        The fish's body. Can be used for set_clip_path().

    Example
    -------

    ```
    fig, ax = plt.subplots()
    bodykwargs=dict(lw=1, edgecolor='k', facecolor='k')
    finkwargs=dict(lw=1, edgecolor='k', facecolor='grey')
    fish = (('Eigenmannia', 'side'), (0, 0), (1, 0), 20.0, -25)
    plot_fish(ax, *fish, bodykwargs=bodykwargs, finkwargs=finkwargs)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    plt.show()
    ```
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
    bpatch = None
    size_fac = 1.1
    bbox = bbox_pathes(*fish.values())
    trans = mpl.transforms.Affine2D()
    angle = np.arctan2(direction[1], direction[0])
    trans.rotate(angle)
    #trans.scale(dxu/dyu, dyu/dxu)   # what is the right scaling????
    trans.scale(1, scaley)
    trans.translate(*pos)
    for part, verts in fish.items():
        if part == 'eye':
            if eyekwargs is not None:
                verts = np.array(verts)*size*size_fac
                verts[:2] = trans.transform_point(verts[:2])
                if not 'zorder' in eyekwargs:
                    eyekwargs['zorder'] = 20
                ax.add_patch(Circle(verts[:2], verts[2], **eyekwargs))
            continue
        verts = bend_path(verts, bend, size, size_fac)
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
        path = path.transformed(trans)
        kwargs = bodykwargs if part == 'body' else finkwargs
        if not 'zorder' in kwargs:
            kwargs['zorder'] = 0 if part == 'body' else 10
        patch = PathPatch(path, **kwargs)
        if part == 'body':
            bpatch = patch
        ax.add_patch(patch)
    return bpatch


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


def plot_pathes(ax, *vertices, **kwargs):
    """ Plot pathes.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the fish.
    vertices: one or more 2D arrays
        The coordinates of pathes to be plotted
        (first column x-coordinates, second colum y-coordinates).
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


def fish_surface(fish, pos=(0, 0), direction=(1, 0), size=20.0, bend=0,
                 gamma=1.0):
    """ Generate meshgrid of one side of the fish from shape.
    
    Parameters
    ----------
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
    gamma: float
        Gamma distortion of the ellipse. The ellipse equation is raised
        to the power of gamma before its smaller diameter is scaled up
        from one to the actual value.

    Returns
    -------
    xx: 2D array of floats
        x-coordinates in direction of body axis.
    yy: 2D array of floats
        y-coordinates in direction upwards from body axis.
    zz: 2D array of floats
        z-coordinates of fish surface, outside of fish NaN.
    """
    if direction[1] != 0:
        raise ValueError('rotation not supported by fish_surface yet.')
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
    path = bend_path(fish['body'], bend, size, size_fac)
    # split in top and bottom half:
    minxi = np.argmin(path[:,0])
    maxxi = np.argmax(path[:,0])
    i0 = min(minxi, maxxi)
    i1 = max(minxi, maxxi)
    path0 = path[i0:i1,:]
    path1 = np.vstack((path[i0::-1,:], path[:i1:-1,:]))
    if np.mean(path0[:,1]) < np.mean(path1[:,1]):
        path0, path1 = path1, path0
    # make sure x coordinates are monotonically increasing:
    pm = np.maximum.accumulate(path0[:,0])
    path0 = np.delete(path0, np.where(path0[:,0] < pm)[0], axis=0)
    pm = np.maximum.accumulate(path1[:,0])
    path1 = np.delete(path1, np.where(path1[:,0] < pm)[0], axis=0)
    # rotate: XXX
    # translate:
    minx = path[minxi,0] + pos[0]
    maxx = path[maxxi,0] + pos[0]
    path0 += pos[:2]
    path1 += pos[:2]
    # interpolate:
    n = 5*max(len(path0), len(path1))
    #n = 200
    x = np.linspace(minx, maxx, n)
    upperpath = np.zeros((len(x), 2))
    upperpath[:,0] = x
    upperpath[:,1] = np.interp(x, path0[:,0], path0[:,1])
    lowerpath = np.zeros((len(x), 2))
    lowerpath[:,0] = x
    lowerpath[:,1] = np.interp(x, path1[:,0], path1[:,1])
    # ellipse origin and semi axes:
    midline = np.array(upperpath)
    midline[:,1] = np.mean(np.vstack((upperpath[:,1], lowerpath[:,1])), axis=0)
    diamy = upperpath[:,1] - midline[:,1]
    diamz = 0.3*diamy  # take it from the top view!
    # apply ellipse:
    y = np.linspace(np.min(midline[:,1]-diamy), np.max(midline[:,1]+diamy), n//2)
    xx, yy = np.meshgrid(x ,y)
    zz = diamz * (np.sqrt(1.0 - ((yy-midline[:,1])/diamy)**2))**gamma
    return xx, yy, zz


def surface_normals(xx, yy, zz):
    """ Normal vectors on a surface.

    Compute surface normals on a surface as returned by `fish_surface()`.

    Parameters
    ----------
    xx: 2D array of floats
        Mesh grid of x coordinates.
    yy: 2D array of floats
        Mesh grid of y coordinates.
    zz: 2D array of floats
        z-coordinates of surface on the xx and yy coordinates.

    Returns
    -------
    nx: 2D array of floats
        x-coordinates of normal vectors for each point in xx and yy.
    ny: 2D array of floats
        y-coordinates of normal vectors for each point in xx and yy.
    nz: 2D array of floats
        z-coordinates of normal vectors for each point in xx and yy.
    """
    dx = xx[0,1] - xx[0,0]
    dy = yy[1,0] - yy[0,0]
    nx = np.zeros(xx.shape)
    nx[:,:-1] = -np.diff(zz, axis=1)/dx
    ny = np.zeros(xx.shape)
    ny[:-1,:] = -np.diff(zz, axis=0)/dy
    nz = np.ones(xx.shape)
    norm = np.sqrt(nx*nx+ny*ny+1)
    return nx/norm, ny/norm, nz/norm


def extract_path(data):
    """ Convert SVG coordinates to numpy array with path coordinates.

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
        The coordinates of pathes
        (first column x-coordinates, second colum y-coordinates).

    Returns
    -------
    bbox: 2D array
        Bounding box of the pathes: [[x0, y0], [x1, y1]]
    """
    # get bounding box of all pathes:
    bbox = np.zeros((2, 2))
    first = True
    for verts in vertices:
        if len(verts.shape) != 2:
            continue
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


def translate_pathes(dx, dy, *vertices):
    """ Translate pathes in place.

    Parameters
    ----------
    dx: float
        Shift in x direction.
    dy: float
        Shift in y direction.
    vertices: one or more 2D arrays
        The coordinates of pathes to be translated
        (first column x-coordinates, second colum y-coordinates).
    """
    for verts in vertices:
        verts[:,0] += dx
        verts[:,1] += dy


def center_pathes(*vertices):
    """ Translate pathes to their common origin in place.

    Parameters
    ----------
    vertices: one or more 2D arrays
        The coordinates of pathes to be centered
        (first column x-coordinates, second colum y-coordinates).
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
        The coordinates of pathes to be rotated
        (first column x-coordinates, second colum y-coordinates).
    """
    theta *= np.pi/180.0
    # rotation matrix:
    c = np.cos(theta)
    s = np.sin(theta)
    rm = np.array(((c, -s), (s, c)))
    # rotation:
    for verts in vertices:
        verts[:,:] = np.dot(verts, rm)


def flipx_pathes(*vertices):
    """ Flip pathes in x-direction in place.

    Parameters
    ----------
    vertices: one or more 2D arrays
        The coordinates of pathes to be flipped
        (first column x-coordinates, second colum y-coordinates).
    """
    for verts in vertices:
        verts[:,0] = -verts[:,0]


def flipy_pathes(*vertices):
    """ Flip pathes in y-direction in place.

    Parameters
    ----------
    vertices: one or more 2D arrays
        The coordinates of pathes to be flipped
        (first column x-coordinates, second colum y-coordinates).
    """
    for verts in vertices:
        verts[:,1] = -verts[:,1]


def mirror_path(vertices1):
    """ Complete path of half a fish outline by appending the mirrored path.

    It is sufficient to draw half of a top view of a fish. Import with
    extract_path() and use this function to add the missing half of the
    outline to the path. The outline is mirrored on the x-axis.

    Parameters
    ----------
    vertices1: 2D array
        The coordinates of one half of the outline of a fish
        (first column x-coordinates, second colum y-coordinates).

    Returns
    -------
    vertices: 2D array
        The coordinates of the complete outline of a fish.
    """
    vertices2 = np.array(vertices1[::-1,:])
    vertices2[:,1] *= -1
    vertices = np.concatenate((vertices1, vertices2))
    return vertices


def normalize_path(*vertices):
    """ Normalize and shift path in place.

    The path extent in x direction is normalized to one and its center
    is shifted to the origin.

    Parameters
    ----------
    vertices: one or more 2D arrays
        The coordinates of the outline of a fish
        (first column x-coordinates, second colum y-coordinates).
    """
    bbox = bbox_pathes(*vertices)
    for verts in vertices:
        verts[:,1] -= np.mean(bbox[:,1])
        verts[:,0] -= bbox[0,0]
        verts /= bbox[1,0] - bbox[0,0]
        verts[:,0] -= 0.5


def bend_path(path, bend, size, size_fac=1.0):
    """ Bend and scale a path.

    Parameters
    ----------
    path: 2D array
        The coordinates of a path.
    bend: float
        Angle for bending in degrees.
    size: float
        Scale path to this size.
    size_fac: float
        Scale path even more, but keep size for calculating the bending.

    Returns
    -------
    path: 2D array
        The coordinates of the bent and scaled path.
    """
    path = np.array(path)
    path *= size_fac*size
    if np.abs(bend) > 1.e-8:
        sel = path[:,0]<0.0
        xp = path[sel,0]   # all negative x coordinates of path
        yp = path[sel,1]   # y coordinates of all negative x coordinates of path
        r = -180.0*0.5*size/bend/np.pi        # radius of circle on which to bend the tail
        beta = xp/r                           # angle on circle for each y coordinate
        R = r-yp                              # radius of point
        path[sel,0] = -np.abs(R*np.sin(beta)) # transformed x coordinates
        path[sel,1] = r-R*np.cos(beta)        # transformed y coordinates
    return path
        

def export_path(vertices):
    """ Print coordinates of path for import as numpy array.

    The variable name, a leading 'np.array([' and the closing '])'
    are not printed.

    Parameters
    ----------
    vertices: 2D array
        The coordinates of the path
        (first column x-coordinates, second colum y-coordinates).
    """
    n = 2
    for k, v in enumerate(vertices):
        if k%n == 0:
            print('   ', end='')
        print(' [%.8e, %.8e],' % (v[0], v[1]), end='')
        if k%n == n-1 and k < len(vertices)-1:
            print('')


def export_fish(name, body, *fins):
    """ Serialize coordinates of fish outlines as a dictionary.

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
        The coordinates of fish's body
        (first column x-coordinates, second colum y-coordinates).
    fins: zero or more 2D arrays
        The coordinates of the fish's fins
        (first column x-coordinates, second colum y-coordinates).

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
    # copy the path specification from an SVG object:
    data = "m 84.013672,21.597656 0.0082,83.002434 0.113201,-0.0145 0.1238,-0.32544 0.06532,-0.80506 0.06836,-0.87696 0.0332,-4.298823 v -8.625 l 0.06836,-1.724609 0.06836,-1.722657 0.07032,-1.726562 0.06836,-1.726563 0.06641,-1.693359 0.03439,-1.293583 0.06912,-1.30798 0.10547,-1.724609 0.10156,-1.724609 0.10352,-1.726563 0.10352,-1.724609 0.13867,-1.72461 0.171876,-2.572265 0.13672,-1.72461 0.13672,-1.726562 0.10352,-1.724609 0.06836,-1.722657 0.103515,-2.574219 0.06836,-1.722656 0.10352,-1.728515 0.07032,-1.722657 0.06836,-1.724609 0.240234,-1.724609 0.34375,-1.72461 0.134766,-1.726562 0.10352,-1.69336 0.03516,-0.875 0.07031,-1.728515 v -0.847657 l -0.07273,-2.246267 -0.0172,-0.184338 0.15636,0.09441 0.384252,1.019739 0.748821,0.905562 1.028854,0.647532 1.356377,-0.03149 0.362644,-0.347764 -0.264138,-0.736289 -1.268298,-1.126614 -1.363988,-0.922373 -0.927443,-0.451153 -0.228986,-0.07018 -0.0015,-0.21624 0.03663,-0.660713 0.480469,-0.847657 -0.101563,-0.876953 -0.103515,-0.845703 -0.103516,-0.876953 -0.207031,-1.695313 -0.273438,-1.724609 -0.308594,-1.726562 -0.27539,-1.72461 -0.310547,-1.722656 -0.240234,-0.878906 -0.400196,-0.877344 -0.53927,-0.596268 -0.486573,-0.216683 z"
    verts = extract_path(data)
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
    # mirror, normalize and export path:
    verts = mirror_path(verts)
    normalize_path(verts)
    fish = export_fish('Alepto_top', verts)
    # plot outline:
    fig, ax = plt.subplots()
    plot_fish(ax, fish, size=1.0/1.1,
              bodykwargs=dict(lw=1, edgecolor='k', facecolor='r'),
              finkwargs=dict(lw=1, edgecolor='k', facecolor='b'))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1)
    plt.show()

    
def main():
    """ Plot some fish shapes and surface normals.
    """
    bodykwargs=dict(lw=1, edgecolor='k', facecolor='none')
    finkwargs=dict(lw=1, edgecolor='k', facecolor='grey')
    eyekwargs=dict(lw=1, edgecolor='white', facecolor='grey')
    var = ['zz', 'nx', 'ny', 'nz']
    fig, ax = plt.subplots()
    for k in range(4):
        y = (1.5-k)*9
        fish = (('Alepto_male', 'side'), (0, y), (1, 0), 20.0, 0)
        xx, yy, zz = fish_surface(*fish, gamma=0.5)
        nx, ny, nz = surface_normals(xx, yy, zz)
        a = [zz, nx, ny, nz]
        th = np.nanmax(np.abs(a[k]))
        ax.contourf(xx[0,:], yy[:,0], -a[k], 20, vmin=-th, vmax=th, cmap='RdYlBu')
        plot_fish(ax, *fish, bodykwargs=bodykwargs, finkwargs=finkwargs, eyekwargs=eyekwargs)
        ax.text(-11, y+2, var[k])
    fish = (('Alepto_male', 'side'), (20, -9), (1, 0), 23.0, 10)
    xx, yy, zz = fish_surface(*fish, gamma=0.8)
    nv = surface_normals(xx, yy, zz)
    ilumn = [-0.05, 0.1, 1.0]
    dv = np.zeros(nv[0].shape)
    for nc, ic in zip(nv, ilumn):
        dv += nc*ic
    #ax.contourf(xx[0,:], yy[:,0], dv, 20, cmap='gist_gray')
    ax.contourf(xx[0,:], yy[:,0], dv, levels=[np.nanmin(dv), np.nanmin(dv)+0.99*(np.nanmax(dv)-np.nanmin(dv)), np.nanmax(dv)], cmap='gist_gray')
    plot_fish(ax, *fish, bodykwargs=bodykwargs, finkwargs=finkwargs, eyekwargs=eyekwargs)
    bodykwargs=dict(lw=1, edgecolor='k', facecolor='k')
    fish = (('Alepto', 'top'), (23, 0), (2, 1), 16.0, -25)
    plot_fish(ax, *fish, bodykwargs=bodykwargs, finkwargs=finkwargs)
    fish = (('Eigenmannia', 'top'), (23, 8), (1, 0.3), 16.0, -15)
    plot_fish(ax, *fish, bodykwargs=bodykwargs, finkwargs=finkwargs)
    fish = (('Eigenmannia', 'side'), (20, 18), (1, 0), 20.0, -25)
    plot_fish(ax, *fish, bodykwargs=bodykwargs, finkwargs=finkwargs, eyekwargs=eyekwargs)
    ax.set_xlim(-15, 35)
    ax.set_ylim(-20, 24)
    plt.show()


if __name__ == '__main__':
    #export_fish_demo()
    main()
    
