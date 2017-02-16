"""
User input from console.

read(): read a single value from console.
select(): select a menue option.
"""

try:
    input_ = raw_input
except NameError:
    input_ = input

    
def read(prompt, default=None, dtype=str, min=None, max=None):
    """
    Read a single input value from the console.
    
    Parameters
    ----------
    prompt: string
        prompt to be displayed.
    default: string
        default value if only 'return' is pressed.
    dtype: type
        data type to be returned (str, int, float, ...)
    min: dtype
        input needs to be larger than min.
    min: dtype
        input needs to be smaller than max.

    Returns
    -------
    x: dtype
        the value of the input.
    """
    
    if default is not None:
        prompt += ' (%s): ' % default
    while True:
        s = input_(prompt)
        if len(s) == 0 and default is not None:
            s = default
        if len(s) > 0:
            try:
                x = dtype(s)
            except ValueError:
                x = None
            if x is not None:
                if min is not None and x < min:
                    continue
                if max is not None and x > max:
                    continue
                return x

            
def select(prompt, default, options, descriptions):
    """
    Print a menue from which the user can select an entry.
    
    Parameters
    ----------
    prompt: string
        A title for the menue.
    default: string
        The default selection.
    options: list of single character strings
        The characters by which the menue options are selected.
    descriptions: list of strings
        A description for each menue option.

    Returns
    -------
    s: string
        the selected value (one of the characters in options).
    """
    print(prompt)
    for o, d in zip(options, descriptions):
        print('  [%s] %s' % (o, d))
    sprompt = '  Select'
    if default is not None:
        sprompt += ' (%s): ' % default
    while True:
        s = input_(sprompt).lower()
        if len(s) == 0:
            s = default
        if s in options:
            return s


            
if __name__ == '__main__':
    x = read('Give me a number between 1 and 10', '5', int, 1, 10)
    print(x)
    print('')
    
    y = select('Your options are', 'a', ['a', 'b', 'o'], ['apples', 'bananas', 'oranges'])
    print(y)
    print('')
    
    
