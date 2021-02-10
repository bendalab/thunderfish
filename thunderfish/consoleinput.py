"""
User input from console.

- `read()`: read a single value from console.
- `select()`: select a menue option.
- `save_inputs()`: write all inputs from `read()` and `select()` into a file.
- `recorded_inputs`: list of strings with all inputs received by `read()` and `select()`.
"""

try:
    input_ = raw_input
except NameError:
    input_ = input


recorded_inputs = []

    
def read(prompt, default=None, dtype=str, min=None, max=None):
    """ Read a single input value from the console.
    
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
        prompt += ' [%s]: ' % default
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
                recorded_inputs.append(s)
                return x

            
def select(prompt, default, options, descriptions):
    """ Print a menue from which the user can select an entry.
    
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
        sprompt += ' [%s]: ' % default
    while True:
        s = input_(sprompt).lower()
        if len(s) == 0:
            s = default
        if s in options:
            recorded_inputs.append(s)
            return s


def save_inputs(file):
    """ Write all inputs from `read()` and `select()` into a file.

    This file can then be used to pipe these inputs to the program
    instead of typing them in manually.
    
    Parameters
    ----------
    file: string
        Name of the file where to save the inputs.
    """
    with open(file, 'w') as df:
        for line in recorded_inputs:
            df.write(line)
            df.write('\n')
            
            
if __name__ == '__main__':
    x = read('Give me a number between 1 and 10', '5', int, 1, 10)
    print(x)
    print('')
    
    y = select('Your options are', 'a', ['a', 'b', 'o'], ['apples', 'bananas', 'oranges'])
    print(y)
    print('')

    print('your successfull inputs have been:')
    print(recorded_inputs)
    
    # save_inputs('test.txt')
    ## you then can call the script like this:
    ## python -m thunderfish.consoleinput < test.txt
