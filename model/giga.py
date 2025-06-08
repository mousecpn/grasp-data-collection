import torch
import torch.nn as nn
from model.decoder import LocalDecoder, BilinearSampler
import torch.nn.functional as F
import time
from utils.common import normalize_3d_coordinate, coordinate2index, normalize_coordinate
from torch_scatter import scatter_mean
from model.unet import UNet

class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    
    '''

    def __init__(self, dim=3, c_dim=128, unet=False, plane_resolution=40, plane_type=['xy', 'xz', 'yz'], kernel_size=3, padding=0.0):
        super().__init__()
        self.actvn = nn.functional.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, start_filts=32, depth=3, )
        else:
            self.unet = None


        self.c_dim = c_dim

        self.reso_plane = plane_resolution

        self.plane_type = plane_type
        self.padding = padding
        # self.pointnet = PointNetPlusPlus(c_dim=c_dim)
        # self.equi3d = EquivariantVoxelEncoderCyclic(obs_channel=1, n_out=c_dim//4, N=4)


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid


    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)

        # Acquire voxel-wise feature
        x = x.unsqueeze(1)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')
                # fea['yz'] = c.permute(0,2,1).reshape(-1,32,40,40,40).mean(0)

        return fea

class GIGA(nn.Module):
    def __init__(self):
        super(GIGA, self).__init__()
        self.encoder = LocalVoxelEncoder(c_dim=32, unet=True, plane_resolution=40)
        self.decoder_qual = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler())
        self.decoder_width = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler())
        self.decoder_rot = LocalDecoder(dim=3, c_dim=96, out_dim=4,  hidden_size=32, feature_sampler=BilinearSampler())
        self.decoder_tsdf = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler())    

        
    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        
        c = self.encoder(inputs)
        qual = self.decoder_qual(p, c)#.sigmoid()
        width = self.decoder_width(p, c)
        rot = self.decoder_rot(p, c)
        rot = nn.functional.normalize(rot, dim=-1)
        
        if p_tsdf is not None:
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width
    

