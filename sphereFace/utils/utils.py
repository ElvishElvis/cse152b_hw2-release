


def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time


# from utils.utils import tb_scalar_dict
def tb_scalar_dict(writer, scalar_dict, iter, task='training'):
    for element in list(scalar_dict):
        obj = scalar_dict[element]
        writer.add_scalar(task + '-' + element, obj, iter)