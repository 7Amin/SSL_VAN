from commons.models.van import VAN
from commons.models.van_v2 import VANV2
from commons.models.van_v3 import VANV3
from commons.models.van_v4 import VANV4
from commons.models.van_v4_1 import VANV41
from commons.models.van_v4_11 import VANV411
from commons.models.van_v4_12 import VANV412
from commons.models.van_v4gl import VANV4GL
from commons.models.van_v5gl import VANV5GL
from commons.models.van_v6gl import VANV6GL
from commons.models.van_v4gl_v1 import VANV4GLV1
from commons.models.van_v4gl_v2 import VANV4GLV2
from commons.models.van_v4_121gl import VANV4121GL
from commons.models.van_v4_122gl import VANV4122GL
from commons.models.unet_p_p import UNetPlusPlus
from commons.models.basic_unet_plus_plus import BasicUNetPlusPlus
from commons.models.seg_res_net_vae import SegResNetVAE
from commons.models.attention_unet import AttentionUnet
from monai.networks.nets import SwinUNETR, UNETR, DynUNet, DiNTS, TopologyInstance, TopologySearch
# from monai.networks.nets import BasicUNetPlusPlus AttentionUnet

from commons.models.pre_training.pre_van_v4 import PREVANV4
from commons.models.pre_training.pre_van_v4gl import PREVANV4GL
from commons.models.pre_training.pre_van_v5gl import PREVANV5GL
from commons.models.pre_training.pre_van_v6gl import PREVANV6GL

from commons.optimizer import get_optimizer

import torch
import os


def get_model(args):
    if args.model_v == "DiNTS_Instance":
        topol = TopologyInstance()
        model = DiNTS(dints_space=topol,
                      in_channels=args.in_channels,
                      num_classes=args.out_channels
                      )

        return model

    if args.model_v == "DiNTS_Search":
        topol = TopologySearch()
        model = DiNTS(dints_space=topol,
                      in_channels=args.in_channels,
                      num_classes=args.out_channels
                      )

        return model

    if args.model_v == "BasicUNetPlusPlus":
        model = BasicUNetPlusPlus(in_channels=args.in_channels,
                                  out_channels=args.out_channels
                                  )

        return model

    if args.model_v == "Unetpp":
        model = UNetPlusPlus(in_channels=args.in_channels,
                             out_channels=args.out_channels
                             )
        return model

    if args.model_v == "AttentionUnet":
        model = AttentionUnet(spatial_dims=3,
                              in_channels=args.in_channels,
                              out_channels=args.out_channels,
                              channels=[256, 256, 512, 512],
                              strides=[2, 2, 2, 2],
                              )
        return model

    if args.model_v == "SegResNetVAE":
        model = SegResNetVAE((args.roi_x, args.roi_y, args.roi_z),
                             vae_estimate_std=False,
                             vae_default_std=0.3,
                             vae_nz=256,
                             spatial_dims=3,
                             init_filters=8,
                             in_channels=args.in_channels,
                             out_channels=args.out_channels,
                             )
        return model

    if args.model_v == "nnUNet":
        strides = [[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
        model = DynUNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            # deep_supervision=True,
            # deep_supr_num=1,
        )
        return model

    if args.model_v == "UNETR16":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0
        )
        return model

    if args.model_v == "UNETR32":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=32,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0
        )
        return model

    if args.model_v == "SwinUNETR48":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            # use_checkpoint=True,
        )
        return model

    if args.model_v == "SwinUNETR36":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=36,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            # use_checkpoint=True,
        )
        return model

    if args.model_v == "SwinUNETR24":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=24,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            # use_checkpoint=True,
        )
        return model

    if args.model_v == "VANV6GL":
        model = VANV6GL(embed_dims=args.embed_dims,
                        mlp_ratios=args.mlp_ratios,
                        depths=args.depths,
                        num_stages=args.num_stages,
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        dropout_path_rate=args.dropout_path_rate,
                        upsample=args.upsample,
                        patch_count=args.patch_count)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "VANV5GL":
        model = VANV5GL(embed_dims=args.embed_dims,
                        mlp_ratios=args.mlp_ratios,
                        depths=args.depths,
                        num_stages=args.num_stages,
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        dropout_path_rate=args.dropout_path_rate,
                        upsample=args.upsample,
                        patch_count=args.patch_count)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "VANV4GLV2":
        model = VANV4GLV2(embed_dims=args.embed_dims,
                          mlp_ratios=args.mlp_ratios,
                          depths=args.depths,
                          num_stages=args.num_stages,
                          in_channels=args.in_channels,
                          out_channels=args.out_channels,
                          dropout_path_rate=args.dropout_path_rate,
                          upsample=args.upsample,
                          patch_count=args.patch_count)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "VANV4GLV1":
        model = VANV4GLV1(embed_dims=args.embed_dims,
                          mlp_ratios=args.mlp_ratios,
                          depths=args.depths,
                          num_stages=args.num_stages,
                          in_channels=args.in_channels,
                          out_channels=args.out_channels,
                          dropout_path_rate=args.dropout_path_rate,
                          upsample=args.upsample,
                          patch_count=args.patch_count)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "VANV4GL":
        model = VANV4GL(embed_dims=args.embed_dims,
                        mlp_ratios=args.mlp_ratios,
                        depths=args.depths,
                        num_stages=args.num_stages,
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        dropout_path_rate=args.dropout_path_rate,
                        upsample=args.upsample,
                        patch_count=args.patch_count)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "VANV41":
        model = VANV41(embed_dims=args.embed_dims,
                       mlp_ratios=args.mlp_ratios,
                       depths=args.depths,
                       num_stages=args.num_stages,
                       in_channels=args.in_channels,
                       out_channels=args.out_channels,
                       dropout_path_rate=args.dropout_path_rate,
                       upsample=args.upsample)
        return model

    if args.model_v == "VANV411":
        model = VANV411(embed_dims=args.embed_dims,
                        mlp_ratios=args.mlp_ratios,
                        depths=args.depths,
                        num_stages=args.num_stages,
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        dropout_path_rate=args.dropout_path_rate,
                        upsample=args.upsample)
        return model

    if args.model_v == "VANV412":
        model = VANV412(embed_dims=args.embed_dims,
                        mlp_ratios=args.mlp_ratios,
                        depths=args.depths,
                        num_stages=args.num_stages,
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        dropout_path_rate=args.dropout_path_rate,
                        upsample=args.upsample)
        return model

    if args.model_v == "VANV4121GL":
        model = VANV4121GL(embed_dims=args.embed_dims,
                           mlp_ratios=args.mlp_ratios,
                           depths=args.depths,
                           num_stages=args.num_stages,
                           in_channels=args.in_channels,
                           out_channels=args.out_channels,
                           dropout_path_rate=args.dropout_path_rate,
                           upsample=args.upsample)
        return model

    if args.model_v == "VANV4122GL":
        model = VANV4122GL(embed_dims=args.embed_dims,
                           mlp_ratios=args.mlp_ratios,
                           depths=args.depths,
                           num_stages=args.num_stages,
                           in_channels=args.in_channels,
                           out_channels=args.out_channels,
                           dropout_path_rate=args.dropout_path_rate,
                           upsample=args.upsample)
        return model

    if args.model_v == "VANV4":
        model = VANV4(embed_dims=args.embed_dims,
                      mlp_ratios=args.mlp_ratios,
                      depths=args.depths,
                      num_stages=args.num_stages,
                      in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      dropout_path_rate=args.dropout_path_rate,
                      upsample=args.upsample)
        return model

    if args.model_v == "VANV3":
        model = VANV3(embed_dims=args.embed_dims,
                      mlp_ratios=args.mlp_ratios,
                      depths=args.depths,
                      num_stages=args.num_stages,
                      in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      dropout_path_rate=args.dropout_path_rate,
                      upsample=args.upsample)
        return model

    if args.model_v == "VANV2":
        model = VANV2(embed_dims=args.embed_dims,
                      mlp_ratios=args.mlp_ratios,
                      depths=args.depths,
                      num_stages=args.num_stages,
                      in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      dropout_path_rate=args.dropout_path_rate,
                      upsample=args.upsample)
        return model

    if args.model_v == "VAN":
        model = VAN(embed_dims=args.embed_dims,
                    mlp_ratios=args.mlp_ratios,
                    depths=args.depths,
                    num_stages=args.num_stages,
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    dropout_path_rate=args.dropout_path_rate,
                    upsample=args.upsample)
        return model

    return None


def get_pre_trained_model(args):
    if args.model_v == "PREVANV6GL":
        model = PREVANV6GL(embed_dims=args.embed_dims,
                           mlp_ratios=args.mlp_ratios,
                           depths=args.depths,
                           num_stages=args.num_stages,
                           in_channels=args.in_channels,
                           out_channels=args.out_channels,
                           dropout_path_rate=args.dropout_path_rate,
                           upsample=args.upsample,
                           patch_count=args.patch_count,
                           cluster_num=args.cluster_num,
                           class_size=args.class_size,
                           embed_dim=args.embed_dim,
                           x_dim=args.roi_x,
                           y_dim=args.roi_y,
                           z_dim=args.roi_z)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "PREVANV5GL":
        model = PREVANV5GL(embed_dims=args.embed_dims,
                           mlp_ratios=args.mlp_ratios,
                           depths=args.depths,
                           num_stages=args.num_stages,
                           in_channels=args.in_channels,
                           out_channels=args.out_channels,
                           dropout_path_rate=args.dropout_path_rate,
                           upsample=args.upsample,
                           patch_count=args.patch_count,
                           cluster_num=args.cluster_num,
                           class_size=args.class_size,
                           embed_dim=args.embed_dim,
                           x_dim=args.roi_x,
                           y_dim=args.roi_y,
                           z_dim=args.roi_z)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "PREVANV4GL":
        model = PREVANV4GL(embed_dims=args.embed_dims,
                           mlp_ratios=args.mlp_ratios,
                           depths=args.depths,
                           num_stages=args.num_stages,
                           in_channels=args.in_channels,
                           out_channels=args.out_channels,
                           dropout_path_rate=args.dropout_path_rate,
                           upsample=args.upsample,
                           patch_count=args.patch_count,
                           cluster_num=args.cluster_num,
                           class_size=args.class_size,
                           embed_dim=args.embed_dim,
                           x_dim=args.roi_x,
                           y_dim=args.roi_y,
                           z_dim=args.roi_z)
        args.model_v = args.model_v + "_" + str(args.patch_count)
        return model

    if args.model_v == "PREVANV4":
        model = PREVANV4(embed_dims=args.embed_dims,
                         mlp_ratios=args.mlp_ratios,
                         depths=args.depths,
                         num_stages=args.num_stages,
                         in_channels=args.in_channels,
                         out_channels=args.out_channels,
                         dropout_path_rate=args.dropout_path_rate,
                         upsample=args.upsample,
                         cluster_num=args.cluster_num,
                         class_size=args.class_size,
                         embed_dim=args.embed_dim,
                         x_dim=args.roi_x,
                         y_dim=args.roi_y,
                         z_dim=args.roi_z)
        return model

    return None


def load_model(args, model, optimizer, scheduler, best_acc, start_epoch):
    temp = os.path.join(args.logdir, args.final_model_url)
    if args.test_mode:
        temp = os.path.join(args.logdir, args.best_model_url)
    if os.path.isfile(temp):
        checkpoint = torch.load(temp, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if 'optimizer' in checkpoint:
            optimizer = get_optimizer(model, args)
            # optimizer_temp = checkpoint['optimizer']
            # optimizer.load_state_dict(optimizer_temp)
        if 'scheduler' in checkpoint:
            scheduler_temp = checkpoint['scheduler']
            scheduler.load_state_dict(scheduler_temp)

    return model, optimizer, scheduler, best_acc, start_epoch
