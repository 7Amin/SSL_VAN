import torch
from thop import profile
# from commons.model_factory import get_model
from commons.model_factory import get_pre_trained_model as get_model


class Config:
    def __init__(self):
        self.in_channels = 1
        self.out_channels = 14
        self.roi_x = 96
        self.roi_y = 96
        self.roi_z = 96
        # self.embed_dims = [96, 192, 384, 768]
        # self.depths = [3, 3, 24, 3]
        self.embed_dims = [128, 128, 512, 512]
        self.depths = [4, 4, 5, 5]
        self.mlp_ratios = [4, 4, 4, 4]
        self.num_stages = 4
        self.cluster_num = 80
        self.class_size = 500

        # self.embed_dims = [64, 128]
        # self.depths = [4, 4]
        # self.mlp_ratios = [4, 4]
        # self.num_stages = 2
        self.dropout_path_rate = 0.0
        self.upsample = "vae"
        self.patch_count = 2


args = Config()
args.model_v = "PREVANV4121double"
model = get_model(args)
input_tensor = torch.randn((1, 1, 96, 96, 96))
flops, params = profile(model, inputs=(input_tensor,))
print(f"model {args.model_v}")
print(f"FLOPS: {flops}")
print(f"FLOPS G: {flops / (1000 * 1000 * 1000)}")
print(f"PARAMS: {params}")
print(f"PARAMS M: {params / (1000 * 1000)}")

# model VANV4
# FLOPS: 103041455616.0
# FLOPS G: 103.041455616
# PARAMS: 33241384.0
# PARAMS M: 33.241384
