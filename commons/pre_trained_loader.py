import torch
import warnings

from torch.nn.modules.module import _IncompatibleKeys


def load_pre_trained(args, model):
    model_url = args.pretrained_dir + args.pretrained_model_name
    from collections import OrderedDict
    model_dict = torch.load(model_url)
    new_state_dict = OrderedDict()
    state_dict = model_dict["state_dict"]
    warnings.warn(f"new_state_dict len is {len(new_state_dict)} - before")
    for key in list(state_dict.keys()):
        if not ("pre_train_proj" in key):
            layer = state_dict.pop(key)
            # warnings.warn(f"key is {key} and layer is: {layer.shape}")
            model_layer = getattr(model, key, None)
            if layer.shape == model_layer.shape:
                new_state_dict[key] = layer
                if args.freeze == "yes":
                    for name, parameter in model.named_parameters():
                        if name == key:
                            parameter.requires_grad = False
                            break
    warnings.warn(f"new_state_dict len is {len(new_state_dict)} - after")
    load_result = model.load_state_dict(new_state_dict, strict=False)
    warnings.warn(f"{args.model_v} - Using pretrained self-supervised backbone weights !")
    if isinstance(load_result, _IncompatibleKeys):
        matched_keys = sum(k in model.state_dict() for k in load_result.missing_keys)
    else:
        matched_keys = len(load_result)

    warnings.warn(f"Number of matched keys loaded: {matched_keys}")

    args.model_v = args.model_v + "_pre"
    return model
