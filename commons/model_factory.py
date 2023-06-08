from commons.models.van import VAN
from commons.models.van_v2 import VANV2
from commons.models.van_v3 import VANV3
from commons.models.van_v4 import VANV4
from commons.models.van_v4gl import VANV4GL
from commons.models.van_v5gl import VANV5GL
from commons.models.van_v6gl import VANV6GL
from commons.models.van_v4gl_v1 import VANV4GLV1
from commons.models.van_v4gl_v2 import VANV4GLV2

from commons.models.pre_training.pre_van_v4 import PREVANV4
from commons.models.pre_training.pre_van_v4gl import PREVANV4GL
from commons.models.pre_training.pre_van_v5gl import PREVANV5GL
from commons.models.pre_training.pre_van_v6gl import PREVANV6GL


def get_model(args):
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
                           patch_count=args.patch_count)
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
                           patch_count=args.patch_count)
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
                           patch_count=args.patch_count)
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
                         upsample=args.upsample)
        return model

    return None
