import os
import warnings


def fix_outputs_url(args, base_url):
    warnings.warn(f" base_url is {base_url}")
    output_url = os.path.join(args.logdir, "output_" + str(args.test_mode) + "_" + str(args.val_mode))
    warnings.warn(f" output_url 1 is {output_url}")
    warnings.warn(f" exists 1 is {os.path.exists(output_url)}")
    if not os.path.isdir(output_url):
        os.mkdir(output_url)
    output_url = output_url + "/" + base_url
    warnings.warn(f" output_url 2 is {output_url}")
    warnings.warn(f" exists 2 is {os.path.exists(output_url)}")
    if not os.path.exists(output_url):
        os.mkdir(output_url)
    args.output_url = output_url
    args.best_model_url = base_url + "_" + "_best.pt"
    args.final_model_url = base_url + "_" + "_final.pt"
    warnings.warn(f" Best url model is {args.best_model_url}, final model url is {args.final_model_url}")
