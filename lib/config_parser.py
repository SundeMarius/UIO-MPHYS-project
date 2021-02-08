"""
Parses a text file with config variables to a dictionary

Author: M.S.S
"""

def read_config_file(filename, delimiter='='):
    # Open file
    with open(filename, 'r') as f:
        lines = f.readlines()

        output = {}
        # Iterate through file
        for l, line in enumerate(lines):

            if line[0] == '#' or line[0] == '\n':
                continue

            # Remove white space and new line char
            line = line.replace(' ', '').replace('\n', '')

            # Get key/value
            try:
                element = line.split(delimiter)
                if len(element) != 2:
                    raise ValueError(
                    ("line %d has invalid format." 
                    " Use one single delimiter '%c'.")%(l+1, delimiter))
                key, value = element
            except ValueError as v:
                print("Error reading '%s':"%filename,v)
                exit(1)
 
            # Add pair to dict
            output[key] = value

    return output
