def create_channel(channel, **kwargs):
    """
    Creates a channel structure
    :param name:
    :param yUnit:
    :param timeUnit:
    :param t:
    :param y:
    :return:
    """
    num_of_params = len(kwargs)
    if num_of_params == 3:
        channel.name = kwargs['name']
        channel.yUnit = kwargs['yUnit']
        channel.tUnit = kwargs['tUnit']
    elif num_of_params == 5:
        channel.name = kwargs['name']
        channel.yUnit = kwargs['yUnit']
        channel.tUnit = kwargs['tUnit']
        if kwargs.get('time') is not None:
            channel.t = kwargs['time'].T.tolist()[0]
        if kwargs.get('value') is not None:
            channel.y = kwargs['value'].T.tolist()[0]

        if kwargs.get('Wavenumber') is not None:
            channel.Wavenumber = kwargs['Wavenumber'].T.tolist()[0]

        if kwargs.get('Intensity') is not None:
            channel.Intensity = kwargs['Intensity']
