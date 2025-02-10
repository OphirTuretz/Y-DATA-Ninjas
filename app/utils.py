def str2bool(s):
    if isinstance(s, bool):
        return s
    false_dict = ['False', 'false', 'f', 'F', 'n', 'N', 'no', 'No', 0, '0']
    true_dict = ['True', 'true', 't', 'T', 'y', 'Y', 'yes', 'Yes', 1, '1']
    if s in false_dict:
        return False
    if s in true_dict:
        return True
    raise ValueError('This is not a bool')
