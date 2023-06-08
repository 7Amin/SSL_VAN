import torch
import warnings


def load_pre_trained(args, model):
    model_url = args.pretrained_dir + args.pretrained_model_name
    from collections import OrderedDict
    model_dict = torch.load(model_url)
    new_state_dict = OrderedDict()
    state_dict = model_dict["state_dict"]

    if "VANV4GL" in args.model_v or "VANV4GLV1" in args.model_v\
            or "VANV4GLV2" in args.model_v or "VANV5GL" in args.model_v or "VANV6GL" in args.model_v:
        for key in list(state_dict.keys()):
            if "van" in key:
                new_state_dict[key] = state_dict.pop(key)
        model.load_state_dict(new_state_dict, strict=False)
        warnings.warn(f"{args.model_v} - Using pretrained self-supervised backbone weights !")

    elif args.model_v in ['VANV4']:
        for key in list(state_dict.keys()):
            if not ("final_conv1" in key):
                new_state_dict[key] = state_dict.pop(key)
        model.load_state_dict(new_state_dict, strict=False)
        warnings.warn(f"{args.model_v} - Using pretrained self-supervised backbone weights !")

    elif args.model_v in ['VANV2', 'VANV3']:
        for key in list(state_dict.keys()):
            if not ("final_conv2" in key):
                new_state_dict[key] = state_dict.pop(key)
        model.load_state_dict(new_state_dict, strict=False)
        warnings.warn(f"{args.model_v} - Using pretrained self-supervised backbone weights !")

    elif args.model_v in ['VAN']:
        for key in list(state_dict.keys()):
            if "van3d" in key:
                new_state_dict[key] = state_dict.pop(key)
        model.load_state_dict(new_state_dict, strict=False)
        warnings.warn(f"{args.model_v} - Using pretrained self-supervised backbone weights !")
    return model
