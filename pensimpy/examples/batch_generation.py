from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE


def run():
    """
    Basic batch generation example which simulates the Sequential Batch Control.
    :return: batch data and Raman spectra in pandas dataframe
    """
    recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
                   FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                   FG: Recipe(FG_DEFAULT_PROFILE, FG),
                   PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                   DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                   WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                   PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}

    recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
    env = PenSimEnv(recipe_combo=recipe_combo)

    return env.get_batches(random_seed=1, include_raman=False)


if __name__ == "__main__":
    print(run())
