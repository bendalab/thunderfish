import os
from collections import OrderedDict


class ConfigFile:
    """Handling of configuration parameter.

    Configuration parameter have a name (key), a value, a unit and a
    description. New parameter can be added with the add() function.

    Configuration parameter can be further structured in the
    configuration file by inserting section titles via add_section().

    The triple value, unit, description can be retrieved via the name
    of a parameter using the [] operator.

    The value of a configuration parameter is retrieved by value()
    and set by set().

    Values of several configuration parameter can be mapped to
    new names with the map() function. The resulting dictionary can be
    passed as key-word arguments to a function.

    The configuration parameter can be written to a configuration file
    with dump() and loaded from a file with load() and load_files().
    """

    def __init__(self, orig=None):
        if orig is None:
            self.cfg = OrderedDict()
            self.sections = dict()
            self.new_section = None
        else:
            self.cfg = OrderedDict(orig.cfg)
            self.sections = dict(orig.sections)
            self.new_section = None

    def __eq__(self, other):
        """Check whether the parameter and ther values are the same.
        """
        return self.cfg == other.cfg

    def add(self, key, value, unit, description):
        """Add a new parameter to the configuration.

        The description of the parameter is a single string. Newline
        characters are intepreted as new paragraphs.

        Args:
          key (string): the key of the parameter
          value (arbitrary): the value of the parameter
          unit (string): the unit of the parameter value
          description (string): a textual description of the parameter.
        """
        # add a pending section:
        if self.new_section is not None:
            self.sections[key] = self.new_section
            self.new_section = None
        # add configuration parameter:
        self.cfg[key] = [value, unit, description]

    def add_section(self, description):
        """Add a new section to the configuration.

        Args:
          description (string): a textual description of the section
        """
        self.new_section = description

    def __getitem__(self, key):
        """Returns the list [value, unit, description]
        of the configuration parameter key.

        Args:
          key (string): the key of the configuration parameter.

        Returns:
          value: the value of the configuraion parameter.
          unit (string): the unit of the configuraion parameter.
          description (string): the description of the configuraion parameter.
        """
        return self.cfg[key]

    def value(self, key):
        """Returns the value of the configuration parameter defined by key.

        Args:
          key (string): the key of the configuration parameter.

        Returns:
          value: the value of the configuraion parameter.
        """
        return self.cfg[key][0]

    def set(self, key, value):
        """Set the value of the configuration parameter defined by key.

        Args:
          key (string): the key of the configuration parameter.
          value: the new value.
        """
        self.cfg[key][0] = value

    def map(self, mapping):
        """Map the values of the configuration onto new names.
        Use this function to generate key-word arguments
        that can be passed on to functions.

        Args:
          mapping (dict): a dictionary with its keys being the new names
          and its values being the parameter names of the configuration.

        Returns:
          a (dict): a dictionary with the keys of mapping
          and the corresponding values retrieved from the configuration
          using the values from mapping.
        """
        a = {}
        for dest, src in mapping.items():
            a[dest] = self.value(src)
        return a

    def dump(self, filename, header=None, maxline=60):
        """Pretty print configuration into file.

        The description of a configuration parameter is printed out
        right before its key-value pair with an initial comment
        character ('#').  

        Section titles get two comment characters prependend ('##').

        Lines are folded if the character count of parameter
        descriptions or section title exceeds maxline.
        
        A header can be printed initially. This is a simple string that is
        formatted like the section titles.

        Args:
            filename: The name of the file for writing the configuration.
            header (string): A string that is written as an introductory comment into the file.
            maxline (int): Maximum number of characters that fit into a line.
        """

        def write_comment(f, comment, maxline, cs):
            # format comment:
            if len(comment) > 0:
                for line in comment.split('\n'):
                    f.write(cs + ' ')
                    cc = len(cs) + 1  # character count
                    for w in line.strip().split(' '):
                        # line too long?
                        if cc + len(w) > maxline:
                            f.write('\n' + cs + ' ')
                            cc = len(cs) + 1
                        f.write(w + ' ')
                        cc += len(w) + 1
                    f.write('\n')

        with open(filename, 'w') as f:
            if header != None:
                write_comment(f, header, maxline, '##')
            maxkey = 0
            for key in self.cfg.keys():
                if maxkey < len(key):
                    maxkey = len(key)
            for key, v in self.cfg.items():
                # possible section entry:
                if key in self.sections:
                    f.write('\n\n')
                    write_comment(f, self.sections[key], maxline, '##')

                # get value, unit, and comment from v:
                val = None
                unit = ''
                comment = ''
                if hasattr(v, '__len__') and (not isinstance(v, str)):
                    val = v[0]
                    if len(v) > 1:
                        unit = ' ' + v[1]
                    if len(v) > 2:
                        comment = v[2]
                else:
                    val = v

                # next key-value pair:
                f.write('\n')
                write_comment(f, comment, maxline, '#')
                f.write('{key:<{width}s}: {val}{unit:s}\n'.format(
                    key=key, width=maxkey, val=val, unit=unit))

    def load(self, filename):
        """Set values of configuration to values from key-value pairs read in
        from file.

        Args:
            filename: The name of the file from which to read the configuration.
        """
        with open(filename, 'r') as f:
            for line in f:
                # do not process empty lines and comments:
                if len(line.strip()) == 0 or line[0] == '#' or not ':' in line:
                    continue
                key, val = line.split(':', 1)
                key = key.strip()
                if not key in self.cfg:
                    continue
                cv = self.cfg[key]
                vals = val.strip().split(' ')
                if hasattr(cv, '__len__') and (not isinstance(cv, str)):
                    unit = ''
                    if len(vals) > 1:
                        unit = vals[1]
                    if unit != cv[1]:
                        print('unit for %s is %s but should be %s'
                              % (key, unit, cv[1]))
                    if type(cv[0]) == bool:
                        cv[0] = (vals[0].lower() == 'true'
                                 or vals[0].lower() == 'yes')
                    else:
                        cv[0] = type(cv[0])(vals[0])
                else:
                    if type(cv[0]) == bool:
                        self.cfg[key] = (vals[0].lower() == 'true'
                                         or vals[0].lower() == 'yes')
                    else:
                        self.cfg[key] = type(cv)(vals[0])

    def load_files(self, cfgfile, filepath, maxlevel=3, verbose=0):
        """Load configuration from current working directory
        as well as from several levels of a file path.

        Args:
          cfgfile (string): name of the configuration file.
          filepath (string): path of a file. Configuration files are read in from different levels of the expanded path.
          maxlevel (int): Read configuration files from up to maxlevel parent directories.
          verbose (int): if greater than zero, print out from which files configuration has been loaded.
        """

        # load configuration from the current directory:
        if os.path.isfile(cfgfile):
            if verbose > 0:
                print('load configuration %s' % cfgfile)
            self.load(cfgfile)

        # load configuration files from higher directories:
        absfilepath = os.path.abspath(filepath)
        dirs = os.path.dirname(absfilepath).split(os.sep)
        dirs.append('')
        ml = len(dirs) - 1
        if ml > maxlevel:
            ml = maxlevel
        for k in xrange(ml, 0, -1):
            path = os.path.join(*(dirs[:-k] + [cfgfile]))
            if os.path.isfile(path):
                print('load configuration %s' % path)
                self.load(path)
