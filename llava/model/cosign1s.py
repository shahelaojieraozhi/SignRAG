import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .stgcn_layers import Graph, get_stgcn_chain

class CoSign1s(nn.Module):
    def __init__(self, in_channels, split, temporal_kernel, hidden_size, modes, level, adaptive=True) -> None:
        super(CoSign1s,self).__init__()
        self.split = split
        self.graph, A = {}, []
        self.part_num = len(self.split)
        self.in_channels = in_channels
        self.modes = modes #['body','hand21','mouth_8','face']
        from .stgcn_layers import Graph #
        for mode in self.modes: # group-by-group processing + graph structure construction | body, hand21, mouth_8, face 
            self.graph[mode] = Graph(layout=f'custom_{mode}', strategy='distance', max_hop=1) 
            #self.graph['body'].A.shape=(2,9,9)|self.graph['hand21'].A.shape=(2,21,21)|self.graph['mouth_8'].A.shape=(2,8,8)|self.graph['face'].A.shape=(2,18,18)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False)) #to tensor
        self.linear = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True)
        )

        self.gcn_modules = {}
        spatial_kernel_size = A[0].size(0) #2
        from .stgcn_layers import get_stgcn_chain #
        for index, mode in enumerate(self.modes): # build the corresponding ST-GCN for each group; returns each ST-GCN output and the last-layer feature dim during training
            self.gcn_modules[mode], final_dim = get_stgcn_chain(64, level, (temporal_kernel, spatial_kernel_size), A[index].clone(), adaptive)
        self.gcn_modules = nn.ModuleDict(self.gcn_modules)
        self.pool_func = F.avg_pool2d
        self.fusion = nn.Sequential(nn.Linear(final_dim*self.part_num, hidden_size), nn.ReLU(inplace=True)) # Linear + ReLU to fuse the grouped skeleton keypoint features
        self.out_size = hidden_size
        self.final_dim = final_dim
    
    def get_feat_dict(self, feature, len_x):
        index = 0
        feat_list = []
        for mode in self.modes:
            if index == 0:
                start, end = 0, self.split[0]
            else:
                start, end = self.split[index-1], self.split[index]
            if mode == 'hand21':
                hand = self.gcn_modules[mode](torch.cat([feature[:,:,:,start:end], \
                                                      feature[:,:,:,end:self.split[index+1]]]))
                left, right = torch.chunk(hand, 2, dim=0)
                feat_list.append(left)
                feat_list.append(right)
                index += 2
            else:
                feat_list.append(self.gcn_modules[mode](feature[:,:,:,start:end]))
                index += 1
        return torch.cat(feat_list, dim=-1)

    def pooling_stage(self, feature):
        feature_list = []
        for i in range(len(self.split)):
            if i == 0:
                start, end = 0, self.split[0]
            else:
                start, end = self.split[i-1], self.split[i]
            feature_list.append(self.pool_func(feature[:,:,:,start:end], \
                                               (1, end - start)).squeeze(-1))
        return torch.cat(feature_list, dim=1)

    def forward(self, x, len_x):
        if x.shape[3] == 7:
            static = torch.cat([x[:,:,:,0:2], x[:,:,:,6].unsqueeze(-1)], dim=-1) # extract (x, y) and conf, then concatenate along the last dim: ([B,T,77,2]) cat ([B,T,77,1]) -> ([B,T,77,3])
        else:
            static = x
        static = static[:,:,:,:self.in_channels] #in_channels=3
        # linear stage
        static = self.linear(static).permute(0,3,1,2) # B,C,T,N
        feat_list = self.get_feat_dict(static, len_x) #(B,C,T,N)-([4,256,196,77])
        cat_feat = self.pooling_stage(feat_list).transpose(1,2) #B,T,C #([4,196,1280])
        fusion_feat = self.fusion(cat_feat) #([4,196,1024])

        return {
            'cat': cat_feat,
            'fusion': fusion_feat
        }
        