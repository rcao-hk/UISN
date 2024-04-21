import torch
from torch import nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine.modules.resnet_block import BasicBlock

from models.resnet_prob import ResNetBase


class Res16UNetBase(ResNetBase):
    INIT_DIM = 32
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)

    def __init__(self, in_channels, out_channels, D=3):
        super(Res16UNetBase, self).__init__(in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def voxelize(self, x: ME.TensorField):
        raise NotImplementedError()

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        raise NotImplementedError()

    def forward(self, x: ME.TensorField):                  # ([744988, 4]), ([744988, 3])
        out, emb = self.voxelize(x)                        # ([51854, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([51854, 4]), ([51854, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([13440, 4]), ([13440, 32])
        out_p2 = self.block1(out)

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))
        out_p4 = self.block2(out)

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)                          # ([51854, 4]), ([51854, 128])
        out = self.block8(out)                             # ([51854, 4]), ([51854, 96])
        return self.devoxelize(out, x, emb)                # ([744988, 4]), ([744988, 13])


class Res16UNet34C(Res16UNetBase):     # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])



class Res16UNet34CSmall(Res16UNet34C):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)



class Res16UNet34CSmaller(Res16UNet34C):
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet34CMC(Res16UNetBase):     # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])



class Res16UNet34CProb(Res16UNetBase):                     # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def __init__(self, in_channels, out_channels, D=3, max_t=40, logit_norm=False):
        super().__init__(in_channels, out_channels, D)
        self.max_t = int(max_t)
        self.logit_norm = logit_norm

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.mu_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):                  # ([Nr, 3])
        out, emb = self.voxelize(x)                        # ([N, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([N, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([N1, 32])
        out_p2 = self.block1(out)                          # ([N1, 32])

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))  # ([N2, 32])
        out_p4 = self.block2(out)                          # ([N2, 64])

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)      # ([N, 128])
        out = self.block8(out)         # ([N, 96]) < ([N, 128])

        if self.logit_norm:
            # LOGIT-NORMAL SPACE ====== #
            emb_mu = self.mu_head(out)        # ([N, 96]) < ([N, 96])
            emb_mu = ME.SparseTensor(emb_mu.F / torch.norm(emb_mu.F, p=2, dim=1, keepdim=True), coordinate_map_key=emb_mu.coordinate_map_key, coordinate_manager=emb_mu.coordinate_manager)
        else:
            # UNCONSTRAINED SPACE ==== #
            emb_mu = MEF.relu(self.mu_head(out))                         # ([N, 96]) < ([N, 96])

        # SIGMA BRANCH =============== #
        emb_sigma2 = MEF.softplus(self.sigma_head(out))    # ([N, 1]) < ([N, 96])

        # if self.max_t > 0 and self.training:
        if self.max_t > 0:
            emb = self.reparam_trick(emb_mu.F, emb_sigma2.F, self.max_t)       # ([m, N, 96])
        else:
            emb = emb_mu.F[None]                                               # ([1, N, 96])

        out_logits = torch.zeros(emb.shape[0], x.shape[0], self.out_channels, device=x.device)        # ([m, N, 13])
        for i in range(emb.shape[0]):
            emb_ = emb[i]
            emb_ = ME.SparseTensor(emb_, coordinate_map_key=emb_mu.coordinate_map_key, coordinate_manager=emb_mu.coordinate_manager)
            out_ = self.final(emb_)
            out_logits[i] = out_.slice(x).F                                                           # ([Nr, 13])

        return out_logits, emb_mu, emb_sigma2                 # ([m, Nr, 13]), ([N, 96]), ([N, 1])

    def reparam_trick(self, emb_mu, emb_sigma2, max_t):
        """
        emb_mu:     ([N, 96])
        emb_sigma2:  ([N, 1])
        return:     ([m, N, 96])
        """
        emb_mu_ext = emb_mu[None].expand(max_t, *emb_mu.shape)                 # ([m, N, 96])
        emb_sigma = emb_sigma2 * 0.5                                           # ([N, 1])
        emb_sigma_ext = emb_sigma[None].expand(max_t, *emb_sigma.shape)        # ([m, N, 1])
        norm_v = torch.randn_like(emb_mu_ext)                                  # ([m, N, 96])
        emb_mu_sto = emb_mu_ext + norm_v * emb_sigma_ext

        return emb_mu_sto



class Res16UNet34CProbMG(Res16UNetBase):                     # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def __init__(self, in_channels, out_channels, D=3, max_t=40, logit_norm=False):
        super().__init__(in_channels, out_channels, D)
        self.max_t = int(max_t)
        self.logit_norm = logit_norm

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.mu_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        self.sigma_p_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_p_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_p_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):                  # ([Nr, 3])
        out, emb = self.voxelize(x)                        # ([N, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([N, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([N1, 32])
        out_p2 = self.block1(out)                          # ([N1, 32])

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))  # ([N2, 32])
        out_p4 = self.block2(out)                          # ([N2, 64])

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)      # ([N, 128])
        out = self.block8(out)         # ([N, 96]) < ([N, 128])

        if self.logit_norm:
            # LOGIT-NORMAL SPACE ====== #
            emb_mu = self.mu_head(out)        # ([N, 96]) < ([N, 96])
            emb_mu = ME.SparseTensor(emb_mu.F / torch.norm(emb_mu.F, p=2, dim=1, keepdim=True), coordinate_map_key=emb_mu.coordinate_map_key, coordinate_manager=emb_mu.coordinate_manager)
        else:
            # UNCONSTRAINED SPACE ==== #
            emb_mu = MEF.relu(self.mu_head(out))                         # ([N, 96]) < ([N, 96])

        # SIGMA BRANCH =============== #
        emb_sigma2 = MEF.softplus(self.sigma_head(out))    # ([N, 1]) < ([N, 96])
        emb_sigma2_p = MEF.softplus(self.sigma_p_head(out))    # ([N, 1]) < ([N, 96])
        emb_sigma2_tol = emb_sigma2 + emb_sigma2_p

        # if self.max_t > 0 and self.training:
        if self.max_t > 0:
            emb = self.reparam_trick(emb_mu.F, emb_sigma2_tol.F, self.max_t)       # ([m, N, 96])
        else:
            emb = emb_mu.F[None]                                               # ([1, N, 96])

        out_logits = torch.zeros(emb.shape[0], x.shape[0], self.out_channels, device=x.device)        # ([m, N, 13])
        for i in range(emb.shape[0]):
            emb_ = emb[i]
            emb_ = ME.SparseTensor(emb_, coordinate_map_key=emb_mu.coordinate_map_key, coordinate_manager=emb_mu.coordinate_manager)
            out_ = self.final(emb_)
            out_logits[i] = out_.slice(x).F                                                           # ([Nr, 13])

        return out_logits, emb_mu, emb_sigma2_tol                 # ([m, Nr, 13]), ([N, 96]), ([N, 1])

    def reparam_trick(self, emb_mu, emb_sigma2, max_t):
        """
        emb_mu:     ([N, 96])
        emb_sigma2:  ([N, 1])
        return:     ([m, N, 96])
        """
        emb_mu_ext = emb_mu[None].expand(max_t, *emb_mu.shape)                 # ([m, N, 96])
        emb_sigma = emb_sigma2 * 0.5                                           # ([N, 1])
        emb_sigma_ext = emb_sigma[None].expand(max_t, *emb_sigma.shape)        # ([m, N, 1])
        norm_v = torch.randn_like(emb_mu_ext)                                  # ([m, N, 96])
        emb_mu_sto = emb_mu_ext + norm_v * emb_sigma_ext

        return emb_mu_sto



class Res16UNet34CSigma(Res16UNetBase):                     # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def __init__(self, in_channels, out_channels, D=3):
        super().__init__(in_channels, out_channels, D)

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.mu_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        self.sigma_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):                  # ([Nr, 3])
        out, emb = self.voxelize(x)                        # ([N, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([N, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([N1, 32])
        out_p2 = self.block1(out)                          # ([N1, 32])

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))  # ([N2, 32])
        out_p4 = self.block2(out)                          # ([N2, 64])

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)      # ([N, 128])
        out = self.block8(out)         # ([N, 96]) < ([N, 128])

        # ----------------------------- mu branch ---------------------------- #
        emb_mu = out                                       # ([N, 96])
        # --------------------------- sigma branch --------------------------- #
        emb_sigma2 = MEF.softplus(self.sigma_head(out))    # ([N, 1]) < ([N, 64]) < ([N, 96])
        # ---------------------------- seg branch ---------------------------- #
        out_cls = self.final(out).slice(x).F            # ([Nr, 13])

        return out_cls.unsqueeze(0), emb_mu, emb_sigma2                 # ([1, Nr, 13]), ([N, 96]), ([N, 1])



class Res16UNet34CAleatoric(Res16UNetBase):                     # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def __init__(self, in_channels, out_channels, D=3, ac_type='softplus'):
        super().__init__(in_channels, out_channels, D)
        self.ac_type = ac_type

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.mu_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.final_sigma = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):                  # ([Nr, 3])
        out, emb = self.voxelize(x)                        # ([N, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([N, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([N1, 32])
        out_p2 = self.block1(out)                          # ([N1, 32])

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))  # ([N2, 32])
        out_p4 = self.block2(out)                          # ([N2, 64])

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)      # ([N, 128])
        out = self.block8(out)         # ([N, 96]) < ([N, 128])  (feature map)

        # MU BRANCH ================== #
        out_logit = self.final(out).slice(x).F             # ([Nr, num_of_cls]) < ([N, 96])
        # SIGMA BRANCH =============== #
        # out_sigma = self.final_sigma(out).slice(x).F       # ([Nr, num_of_cls]) < ([N, 96])

        out_sigma = self.final_sigma(out)       # ([Nr, num_of_cls]) < ([N, 96])
        if self.ac_type == 'softplus':
            out_sigma = MEF.softplus(out_sigma).slice(x).F                # ([N, 96]) < ([N, 96])
        elif self.ac_type == 'sigmoid':
            out_sigma = MEF.sigmoid(out_sigma).slice(x).F                # ([N, 96]) < ([N, 96])

        return out_logit, out_sigma    # ([Nr, num_of_cls]), ([Nr, num_of_cls])



class Res16UNet34CDUL(Res16UNetBase):                # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def __init__(self, in_channels, out_channels, D=3, ac_type='softplus'):
        super().__init__(in_channels, out_channels, D)
        self.ac_type = ac_type

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.mu_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        # self.mu_head = nn.Sequential(
        #     ME.MinkowskiDropout(p=0.4),
        #     ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        #     ME.MinkowskiBatchNorm(self.PLANES[7] * self.BLOCK.expansion, eps=2e-5),
        # )
        # self.logsigma2_head = nn.Sequential(
        #     ME.MinkowskiDropout(p=0.4),
        #     ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        #     ME.MinkowskiBatchNorm(self.PLANES[7] * self.BLOCK.expansion, eps=2e-5),
        # )

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):                  # ([Nr, 3])
        out, emb = self.voxelize(x)                        # ([N, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([N, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([N1, 32])
        out_p2 = self.block1(out)                          # ([N1, 32])

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))  # ([N2, 32])
        out_p4 = self.block2(out)                          # ([N2, 64])

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)      # ([N, 128])
        out = self.block8(out)         # ([N, 96]) < ([N, 128])  (feature map)

        out_mu = MEF.relu(self.mu_head(out))
        out_sigma2 = MEF.softplus(self.sigma_head(out))
        out_logsigma2 = ME.SparseTensor(torch.log(out_sigma2.F), coordinate_map_key=out_sigma2.coordinate_map_key, coordinate_manager=out_sigma2.coordinate_manager)

        # out_mu = self.mu_head(out)                         # ([N, 96])
        # out_logsigma2 = self.logsigma2_head(out)           # ([N, 96])

        out_emb = self.reparam_trick(out_mu.F, out_logsigma2.F)                # ([N, 96])
        out_emb = ME.SparseTensor(out_emb, coordinate_map_key=out_mu.coordinate_map_key, coordinate_manager=out_mu.coordinate_manager)

        out_logit = self.final(out_emb).slice(x).F         # ([Nr, num_of_cls]) < ([N, 96])

        return out_logit, out_mu, out_logsigma2            # ([Nr, num_of_cls]), ([Nr, num_of_cls])

    def reparam_trick(self, mu, logsigma2):
        """
        mu:     ([N, 96])
        logsigma2:  ([N, 96])
        return:     ([N, 96])
        """

        std = torch.exp(logsigma2).sqrt()                 # ([N, 96])
        epsilon = torch.randn_like(std)                    # ([N, 96])

        return mu + epsilon * std

    def differentiable_sample(self, mu, logsigma2):
        import torch.distributions as d
        distribution = d.Normal(mu, torch.exp(logsigma2))
        mu_hat = distribution.rsample((1,)).squeeze(0)
        return mu_hat



class Res16UNet34CRUL(Res16UNetBase):                # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def __init__(self, in_channels, out_channels, D=3, ac_type='softplus'):
        super().__init__(in_channels, out_channels, D)
        self.ac_type = ac_type

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F                  # ([821442, 4])

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D) # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion                                                    # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion                                                   # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D) # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM                                                                           # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.mu_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.PLANES[7] * self.BLOCK.expansion, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head = nn.Sequential(
            ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, int(self.PLANES[7] * self.BLOCK.expansion / 2), kernel_size=1, stride=1, bias=True, dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(int(self.PLANES[7] * self.BLOCK.expansion / 2), 1, kernel_size=1, stride=1, bias=True, dimension=D),
        )
        self.sigma_head[-1].kernel = torch.nn.parameter.Parameter(torch.zeros_like(self.sigma_head[-1].kernel))
        self.sigma_head[-1].bias = torch.nn.parameter.Parameter(torch.log(torch.tensor([1e-3])))   # NOTE: initialize as suggested

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):                  # ([Nr, 3])
        out, emb = self.voxelize(x)                        # ([N, 4]), None
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))  # ([N, 32])

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))  # ([N1, 32])
        out_p2 = self.block1(out)                          # ([N1, 32])

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))  # ([N2, 32])
        out_p4 = self.block2(out)                          # ([N2, 64])

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)      # ([N, 128])
        out = self.block8(out)         # ([N, 96]) < ([N, 128])  (feature map)

        out_mu = MEF.relu(self.mu_head(out))
        out_sigma2 = MEF.softplus(self.sigma_head(out))
        out_logsigma2 = ME.SparseTensor(torch.log(out_sigma2.F), coordinate_map_key=out_sigma2.coordinate_map_key, coordinate_manager=out_sigma2.coordinate_manager)

        # if self.training:
        #     out_mu_wave, index = self.mix(out_mu.F, out_sigma2.F.mean(dim=1, keepdim=True))
        #     out_emb = ME.SparseTensor(out_mu_wave, coordinate_map_key=out_mu.coordinate_map_key, coordinate_manager=out_mu.coordinate_manager)

        #     index_sparse = ME.SparseTensor(index.unsqueeze(1), coordinate_map_key=out_mu.coordinate_map_key, coordinate_manager=out_mu.coordinate_manager)
        #     index_dense = index_sparse.slice(x).F

        #     out_logit = self.final(out_emb).slice(x).F         # ([Nr, num_of_cls]) < ([N, 96])

        #     return out_logit, index_dense            # ([Nr, num_of_cls]), ([Nr, num_of_cls])
        # else:
        #     out_logit = self.final(out_mu).slice(x).F         # ([Nr, num_of_cls]) < ([N, 96])
        #     return out_logit, out_mu, out_logsigma2

        out_mu_wave, index = self.mix(out_mu.F, out_sigma2.F.mean(dim=1, keepdim=True))
        out_emb = ME.SparseTensor(out_mu_wave, coordinate_map_key=out_mu.coordinate_map_key, coordinate_manager=out_mu.coordinate_manager)

        index_sparse = ME.SparseTensor(index.unsqueeze(1), coordinate_map_key=out_mu.coordinate_map_key, coordinate_manager=out_mu.coordinate_manager)
        index_dense = index_sparse.slice(x).F

        # NOTE this behaviour looks wierd because index_dense is indeed unnecrssary when evaluating, 
        # but it is a workaround to suppress Minkowski CUDA out of memeroy when training
        if self.training:
            out_logit = self.final(out_emb).slice(x).F         # ([Nr, num_of_cls]) < ([N, 96])
        else:
            out_logit = self.final(out_mu).slice(x).F         # ([Nr, num_of_cls]) < ([N, 96])  # NOTE uncommented when training
            # out_logit = self.final(out_mu)         # ([Nr, num_of_cls]) < ([N, 96])   # NOTE uncommented when evaluting

        return out_logit, index_dense, out_logsigma2            # ([Nr, num_of_cls]), ([Nr, num_of_cls])


    def mix(self, mu, sigma2):
        index = torch.randperm(mu.shape[0]).to(mu.device)
        mu_i, sigma2_i = mu, sigma2
        mu_j, sigma2_j = mu_i[index], sigma2_i[index]
        sigma2_hati = sigma2_i / (sigma2_i + sigma2_j)
        sigma2_hatj = sigma2_j / (sigma2_i + sigma2_j)
        mu_wave = sigma2_hati * mu_i + sigma2_hatj * mu_j
        return mu_wave, index


    def reparam_trick(self, mu, logsigma2):
        """
        mu:     ([N, 96])
        logsigma2:  ([N, 96])
        return:     ([N, 96])
        """

        std = torch.exp(logsigma2).sqrt()                 # ([N, 96])
        epsilon = torch.randn_like(std)                    # ([N, 96])

        return mu + epsilon * std
