import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('implicitstainer')
class ImplicitStainer(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=False):
        super().__init__()
        print(imnet_spec)
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 1
            imnet_in_dim += imnet_in_dim # attach coord
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
            self.imnet1 = models.make(imnet_spec, args={'in_dim': 2,'out_dim':self.encoder.out_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        feat = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        bs, q = coord.shape[:2]
        coord = self.imnet1(coord.view(bs * q, -1)).view(bs, q, -1)

        inp = torch.cat([feat, coord], dim=-1)
        ret = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        return ret

    def forward(self, inp, coord):
        self.gen_feat(inp)
        return self.query_rgb(coord)




