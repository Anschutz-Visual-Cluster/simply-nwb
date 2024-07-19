import matplotlib as mpl


# Code sourced from: https://github.com/jbhunt/myphdlib/blob/7a6dd65fa410e985853027767d95010872aff505/myphdlib/extensions/matplotlib.py
def removeArrowKeyBindings():
    """
    """

    pairs = (
        ('back', 'left'),
        ('forward', 'right')
    )
    for action, key in pairs:
        if key in mpl.rcParams[f'keymap.{action}']:
            mpl.rcParams[f'keymap.{action}'].remove(key)

    return
