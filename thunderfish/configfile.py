"""Writing and loading of configuration files.

dump_config(): write dictionary to configuration file.
load_config(): load configuration file to dictionary.
"""

import os

def dump_config(filename, cfg, sections=None, header=None, maxline=60) :
    """
    Pretty print non-nested dicionary cfg into file.

    The keys of the dictionary are strings.
    
    The values of the dictionary can be single variables or lists:
    [value, unit, comment]
    Both unit and comment are optional.

    value can be any type of variable.

    unit is a string (that can be empty).
    
    Comments comment are printed out right before the key-value pair.
    Comments are single strings. Newline characters are intepreted as
    new paragraphs.  Lines are folded if the character count exceeds
    maxline.

    Section comments can be added by the sections dictionary.
    It contains comment strings as values that are inserted right
    before the key-value pair with the same key. Section comments
    are formatted in the same way as comments for key-value pairs,
    but get two comment characters prependend ('##').

    A header can be printed initially. This is a simple string that is
    formatted like the section comments.

    Args:
        filename: The name of the file for writing the configuration.
        cfg (dict): Configuration keys, values, units, and comments.
        sections (dict): Comments describing secions of the configuration file.
        header (string): A string that is written as an introductory comment into the file.
        maxline (int): Maximum number of characters that fit into a line.
    """

    def write_comment( f, comment, maxline=60, cs='#' ) :
        # format comment:
        if len( comment ) > 0 :
            for line in comment.split( '\n' ) :
                f.write( cs + ' ' )
                cc = len( cs ) + 1  # character count
                for w in line.strip().split( ' ' ) :
                    # line too long?
                    if cc + len( w ) > maxline :
                        f.write( '\n' + cs + ' ' )
                        cc = len( cs ) + 1
                    f.write( w + ' ' )
                    cc += len( w ) + 1
                f.write( '\n' )
    
    with open( filename, 'w' ) as f :
        if header != None :
            write_comment( f, header, maxline, '##' )
        maxkey = 0
        for key in cfg.keys() :
            if maxkey < len( key ) :
                maxkey = len( key )
        for key, v in cfg.items() :
            # possible section entry:
            if sections != None and key in sections :
                f.write( '\n\n' )
                write_comment( f, sections[key], maxline, '##' )

            # get value, unit, and comment from v:
            val = None
            unit = ''
            comment = ''
            if hasattr(v, '__len__') and (not isinstance(v, str)) :
                val = v[0]
                if len( v ) > 1 :
                    unit = ' ' + v[1]
                if len( v ) > 2 :
                    comment = v[2]
            else :
                val = v

            # next key-value pair:
            f.write( '\n' )
            write_comment( f, comment, maxline, '#' )
            f.write( '{key:<{width}s}: {val}{unit:s}\n'.format( key=key, width=maxkey, val=val, unit=unit ) )


def load_config(filename, cfg) :
    """Set values of dictionary cfg to values from key-value pairs read in
    from file.
    
    Args:
        filename: The name of the file from which to read the configuration.
        cfg (dict): Configuration keys, values, units, and comments.
    """
    with open( filename, 'r' ) as f :
        for line in f :
            # do not process empty lines and comments:
            if len( line.strip() ) == 0 or line[0] == '#' or not ':' in line :
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            if not key in cfg :
                continue
            cv = cfg[key]
            vals = val.strip().split( ' ' )
            if hasattr(cv, '__len__') and (not isinstance(cv, str)) :
                unit = ''
                if len( vals ) > 1 :
                    unit = vals[1]
                if unit != cv[1] :
                    print 'unit for', key, 'is', unit, 'but should be', cv[1]
                if type(cv[0]) == bool :
                    cv[0] = ( vals[0].lower() == 'true' or vals[0].lower() == 'yes' )
                else :
                    cv[0] = type(cv[0])(vals[0])
            else :
                if type(cv[0]) == bool :
                    cfg[key] = ( vals[0].lower() == 'true' or vals[0].lower() == 'yes' )
                else :
                    cfg[key] = type(cv)(vals[0])


def load_config_files(cfgfile, filepath, cfg, maxlevel=3, verbose=0) :
    """Load configuration from current working directory
    as well as from several levels of a file path.

    Args:
      cfgfile (string): name of the configuration file.
      filepath (string): path of a file. Configuration files are read in from different levels of the expanded path.
      cfg (dict): the configuration data.
      maxlevel (int): Read configuration files from up to maxlevel parent directories.
      verbose (int): if greater than zero, print out from which files configuration has been loaded.
    """
    
    # load configuration from the current directory:
    if os.path.isfile(cfgfile) :
        if verbose > 0 :
            print('load configuration %s' % cfgfile)
        load_config(cfgfile, cfg)

    # load configuration files from higher directories:
    absfilepath = os.path.abspath(filepath)
    dirs = os.path.dirname(absfilepath).split('/')
    dirs.append('')
    ml = len(dirs)-1
    if ml > maxlevel :
        ml = maxlevel
    for k in xrange(ml, 0, -1) :
        path = '/'.join(dirs[:-k]) + '/' + cfgfile
        if os.path.isfile( path ) :
            print('load configuration %s' % path)
            load_config(path, cfg)
