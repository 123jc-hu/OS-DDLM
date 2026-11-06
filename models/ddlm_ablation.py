"""Decoder controls of DDLM-Extractor (rank 8): one component changed at a time."""

from torch import nn
from models.Ours_light import Model as FullModel, SpatialFusionLowRank

VARIANTS = ('late_downsampling', 'no_dilation', 'full_spatial', 'rank2', 'rank4')


class Model(FullModel):
    def __init__(self, config):
        super().__init__(config)
        variant = config['ddlm_ablation']
        conv = self.channel_tcb_block[1]
        fusion = self.spatiotemporal_fusion_block[1]
        if variant == 'late_downsampling':
            conv.stride = (1, 1)
            self.spatiotemporal_fusion_block[2] = nn.AvgPool2d((1, 8))
        elif variant == 'no_dilation':
            receptive_field = (conv.kernel_size[1]-1)*conv.dilation[1]+1
            self.channel_tcb_block[1] = nn.Conv2d(
                conv.in_channels, conv.out_channels, (1, receptive_field),
                stride=conv.stride, padding=(0, receptive_field//2),
                groups=conv.groups, bias=False)
        elif variant == 'full_spatial':
            self.spatiotemporal_fusion_block[1] = nn.Sequential(
                nn.Conv2d(fusion.proj.in_channels, fusion.mix.out_channels,
                          (self.n_channels, 1), bias=False), fusion.bn, fusion.act)
        elif variant in ('rank2', 'rank4'):
            self.spatiotemporal_fusion_block[1] = SpatialFusionLowRank(
                self.n_channels, group_dim=fusion.proj.in_channels, r=int(variant[-1]))
        else:
            raise ValueError(f'Unknown DDLM ablation: {variant}')
