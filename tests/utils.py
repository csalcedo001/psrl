def get_error_msg(got, expected, context=None):
    msg = '\nGot: {}\nExpected: {}'.format(got, expected)
    if context is not None:
        msg = '{}\nContext: {}'.format(msg, context)
    return msg