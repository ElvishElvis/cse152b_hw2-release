
# import logging
# from utils.utils import datasize
def datasize(train_loader, batch_size, tag='train'):
    print('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*batch_size, len(train_loader)))
    pass



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


import torch
# save model
# from utils.utils import save_model
def save_model(save_path, iter, net, optimizer, loss):
    torch.save(
        {
            'iter': iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        save_path)
    return True

# load model
def load_checkpoint(PATH):
    checkpoint = torch.load(PATH)
    return checkpoint
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']