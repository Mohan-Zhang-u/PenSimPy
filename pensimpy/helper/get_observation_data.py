def get_observation_data(observation, t):
    """
    Get observation data at t.
    """
    vars = ['Foil', 'Fw', 'Fs', 'Fa', 'Fb', 'Fc', 'Fh', 'Fg', 'Wt', 'Fremoved', 'DO2', 'pH', 'T',
            'O2', 'pressure']
    return [[var, eval(f"observation.{var}.y[t]", {'observation': observation, 't': t})] for var in vars]
