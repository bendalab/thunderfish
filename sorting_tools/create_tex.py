__author__ = 'raab'
def description_of_code():
    """
    Builds a .pdf data.

    Arg 1: this script it self
    Arg 2: .npy data from wavefish
    Arg 3: .npy data from pulsefish
    Arg 4: str Titel of the PDF
    Arg 5: .pdf figure with distribution of wave and pulsefish
    Arg 6: .pdf figure with distribution of eod frequenies
    Arg 7: .pdf figure with distribution of beat frequencies
    :return:
    """

import numpy as np
import sys
import os
from IPython import embed

def build_tex_pdf(wavefish, pulsefish):
    tf = open( '%s.tex' % sys.argv[3], 'w')
    tf.write( '\\documentclass[a4paper,12pt,pdflatex]{article}\n' )
    tf.write( '\\usepackage{graphics}\n' )
    tf.write( '\n' )
    tf.write( '\\begin{document}\n' )
    tf.write( '\\section*{%s}\n' %sys.argv[3])

    response = raw_input('Do you want to enter extra Data (loation, temp, cond. etc ? yes[y] or no[n]')
    if response == "y":
        tf.write( '\n' )
        tf.write( '\n' )
        tf.write( '\n' )
        tf.write( '\n' )
        tf.write( '\\begin{tabular}[t]{rr}\n' )
        tf.write( '\\hline\n' )
        # tf.write( 'Question & Answer \\\\ \\hline \n' )
        tf.write( '\\\\\n' )

        Country = raw_input('What country ?')
        tf.write( 'Country: & %s\\\\\n' % Country )
        tf.write( '\\\\\n' )

        Location = raw_input('What location ?')
        tf.write( 'Location: & %s\\\\\n' % Location )
        tf.write( '\\\\\n' )

        Date = raw_input('What date ?')
        tf.write( 'Date: & %s\\\\\n' % Date )
        tf.write( '\\\\\n' )

        Water_conductivity = raw_input('What water conductivity ?')
        tf.write( 'Water condutivity: & %s\\\\\n' % Water_conductivity )
        tf.write( '\\\\\n' )

        Water_temp = raw_input('What water temperatur ?')
        tf.write( 'Water temperature: & %s\\\\\n' % Water_temp )
        tf.write( '\\\\\n' )

        tf.write( '\\hline\n' )
        tf.write( '\\end{tabular}\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\\includegraphics{%s}\n' %sys.argv[4])
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\\pagebreak\n' )
    tf.write( '\\includegraphics{%s}\n' %sys.argv[5])
    tf.write( '\\includegraphics{%s}\n' %sys.argv[6])
    tf.write( '\\pagebreak\n' )
    tf.write( '\\section*{Wavefishes list}\n')
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\\begin{tabular}[t]{rr}\n' )
    tf.write( '\\hline\n' )
    tf.write( 'Wavefish no. & freq [Hz] \\\\ \\hline \n' )

    for i in np.arange(len(wavefish)):
        help_var = i+1
        if help_var % 35 == 0:
            tf.write( '%s & %s \\\\\n' % (i+1, wavefish[i]) )
            tf.write( '\\hline\n' )
            tf.write( '\\end{tabular}\n' )
            tf.write( '\\begin{tabular}[t]{rr}\n' )
            tf.write( '\\hline\n' )
            tf.write( 'Wavefish no. & freq [Hz] \\\\ \\hline \n' )

        else:
            tf.write( '%s & %s \\\\\n' % (i+1, wavefish[i]) )

    tf.write( '\\hline\n' )
    tf.write( '\\end{tabular}\n' )
    # tf.write( '\\pagebreak\n' )
    tf.write( '\n' )
    tf.write( '\\section*{Pulsefishes list}\n')
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\\begin{tabular}[t]{rr}\n' )
    tf.write( '\\hline\n' )
    tf.write( 'Pulsefish no. & freq [Hz] \\\\ \\hline \n' )

    for j in np.arange(len(pulsefish)):
        help_var2 = j+1
        if help_var2 % 35 == 0:
            tf.write( '%s & %s \\\\\n' % (j+1, pulsefish[j]) )
            tf.write( '\\hline\n' )
            tf.write( '\\end{tabular}\n' )
            tf.write( '\\begin{tabular}[t]{rr}\n' )
            tf.write( '\\hline\n' )
            tf.write( 'Pulsefish no. & freq [Hz] \\\\ \\hline \n' )

        else:
            tf.write( '%s & %s \\\\\n' % (j+1, pulsefish[j]) )

    tf.write( '\\hline\n' )
    tf.write( '\\end{tabular}\n' )
    tf.write( '\n' )
    tf.write( '\n' )
    tf.write( '\\end{document}\n' )
    tf.close()
    os.system( 'pdflatex %s' %sys.argv[3])
    os.remove( '%s.aux' %sys.argv[3] )
    os.remove( '%s.log' %sys.argv[3])
    os.remove( '%s.tex' %sys.argv[3])

def load_npy_convert_list():
    wavefish = np.load('%s' % sys.argv[1])
    wavefish = wavefish.tolist()
    for i in np.arange(len(wavefish)):
        wavefish[i] = "%.2f" % wavefish[i]

    pulsefish = np.load('%s' % sys.argv[2])
    pulsefish = pulsefish.tolist()
    for j in np.arange(len(pulsefish)):
        pulsefish[j] = "%.2f" % pulsefish[j]
    # embed()

    return wavefish, pulsefish


def main():
    print '### Lets make a .tex file ###'
    wavefish, pulsefish = load_npy_convert_list()
    build_tex_pdf(wavefish, pulsefish)


if __name__ == '__main__':
    main()