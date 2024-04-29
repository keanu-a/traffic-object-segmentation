from torchvision.datasets import Cityscapes

cityscape_classes = Cityscapes.classes


def get_ignored_classes():
    """
    Gets ignored classes from the cityscapes classes file
    """
    ignore = []
    for c in cityscape_classes:
        if c.ignore_in_eval:
            ignore.append(c.id)

    return ignore


def get_used_classes():
    """
    Gets used classes from the cityscapes classes file
    Adds 34 to handle ignored
    """
    used = []
    for c in cityscape_classes:
        if not c.ignore_in_eval:
            used.append(c.id)

    used.append(34)
    return used


def get_used_colors():
    """
    Gets used class colors from the cityscapes classes file
    Adds [0, 0, 0] to handle ignored
    """
    colors = [[0, 0, 0]]
    for c in cityscape_classes:
        if not c.ignore_in_eval:
            color = c.color
            colors.append([color[0], color[1], color[2]])

    return colors
