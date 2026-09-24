import torch
from torch import nn
from models import resnet


def strip_module_prefix(state_dict):
    """
    Strip a leading "module." from state_dict keys.

    Checkpoints saved from a DataParallel-wrapped model (including the
    original MedicalNet pretrained weights, and any checkpoint saved by
    this project before DataParallel was removed for single-GPU runs) have
    this prefix; checkpoints saved from an unwrapped model don't. Applying
    this unconditionally makes loading work either way.
    """
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()}


def generate_model(model_name='resnet10', num_seg_classes=3, no_cuda=False, phase='train', pretrain_path=None, new_layer_names=['avgpool','fc', 'fc_heads','fc_disease']):

    if model_name == 'resnet10':
        model = resnet.resnet10(num_seg_classes=num_seg_classes)
    elif model_name == 'resnet50':
        model = resnet.resnet50(num_seg_classes=num_seg_classes)
    elif model_name == 'resnet18':
        model = resnet.resnet18(num_seg_classes=num_seg_classes) 
    elif model_name == 'resnet101':
        model = resnet.resnet101(num_seg_classes=num_seg_classes) 
    elif model_name == 'resnet152':
        model = resnet.resnet152(num_seg_classes=num_seg_classes) 
    if not no_cuda:
        if torch.cuda.device_count() > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            net_dict = model.state_dict()
        else:
            # Single GPU: skip DataParallel entirely — it has nothing to
            # parallelize across and only adds scatter/gather overhead.
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
            model = model.cuda()
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if phase != 'test' and pretrain_path:
        print ('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)
        pretrain_state = strip_module_prefix(pretrain['state_dict'])
        pretrain_dict = {k: v for k, v in pretrain_state.items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        missing, unexpected = model.load_state_dict(net_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}
        new_ids = {id(p) for p in new_parameters}
        for name, p in model.named_parameters():
            if id(p) in new_ids:
                print("[NEW ]", name, p.shape)
            else:
                print("[BASE]", name, p.shape)

        return model, parameters

    return model, model.parameters()

