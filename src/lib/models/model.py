import torch
from .networks.msfcnet import get_msfcnet
from .networks.msfcnet_resnext import get_msfcnetresnext
from .networks.msfcnet_cspdarknet import get_msfcnetcspdarknet
from .networks.msfcnet_detnet import get_msfcnetdetnet
from .networks.msfcnet_resnet import get_msfcnetresnet
from .networks.msfcnet_vgg import get_msfcnetvgg

_model_factory = {
    'msfc':get_msfcnet,
    'msfcresnext':get_msfcnetresnext,
    'msfcresnet':get_msfcnetresnet,
    'msfcdetnet':get_msfcnetdetnet,
    'msfccspdarknet':get_msfcnetcspdarknet,
    'msfcvgg':get_msfcnetvgg,
}


def create_model(arch , heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:])  if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer = None,resume = False,
               lr = None, lr_step = None):
    start_epoch = 0
    checkpoint = torch.load(model_path , map_location= lambda storage,loc: storage)
    print(('loaded {}, epoch {}'.format(model_path, checkpoint['epoch'])))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None:
        if 'optimizer' in checkpoint:
            if resume ==True:
                start_epoch = 1
            else:
                start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.5

            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr',start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model,optimizer,start_epoch
    else:
        return model

def save_model(path, epoch,model,optimizer = None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
