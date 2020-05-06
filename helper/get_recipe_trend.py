import numpy as np
import itertools


def get_recipe_trend(recipe, recipe_sp):
    """
    Get the recipe trend data
    :param recipe:
    :param recipe_sp:
    :return:
    """
    recipe = [recipe[0]] + np.diff(recipe).tolist()
    recipe_sp = [[ele] for ele in recipe_sp]
    res_default = [x * y for x, y in zip(recipe, recipe_sp)]
    return list(itertools.chain(*res_default))[0:1150]
