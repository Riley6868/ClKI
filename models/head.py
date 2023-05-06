import torch as t
import torch.nn as nn


class Head(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self,patch_num,num_classes):
        super(Head, self).__init__()
        self.avgpools = nn.ModuleList()
        for i in range(4):
            avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1))
            self.avgpools.append(avgpool)

        self.heads = nn.ModuleList()
        for i in range(4):
            head = nn.Sequential(nn.Linear(patch_num,num_classes))
            self.heads.append(head)



    def forward(self, out_list,x0):
        out_fea = []
        out_pred = []
        #4
        for i in range(4):
            x = self.avgpools[i](out_list[i])
            x = t.flatten(x, 1)
            x = t.cat((x0,x),1)
            out_fea.append(x)
            #x = out_list[i][:,0,:]
           # out_fea.append(x)
            x = self.heads[i](x)
            out_pred.append(x)


        return out_pred,out_fea