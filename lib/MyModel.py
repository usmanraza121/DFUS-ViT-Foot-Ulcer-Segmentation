from .pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .modules import*
from torchsummary import summary
"""Main model"""
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# ----------------
class conv_2nV1(nn.Module):
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super(conv_2nV1, self).__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        self.relu = nn.ReLU(True)
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        if self.main == 0:
            # stage 2
            self.h2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_hc, out_c, 1)

        elif self.main == 1:
            # stage 2
            self.h2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_lc, out_c, 1)

        else:
            raise NotImplementedError

    def forward(self, in_h, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = self.relu(self.bnh_1(h2h + l2h))
        l = self.relu(self.bnl_1(l2l + h2l))

        if self.main == 0:
            # stage 2
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))
            h_fuse = self.relu(self.bnh_2(h2h + l2h))

            # stage 3
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
            # 这里使用的不是in_h，而是h
        elif self.main == 1:
            # stage 2
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))

            # stage 3
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
        else:
            raise NotImplementedError

        return out
####################################
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x
class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)
        """can ope these comments as I forget to comment when I trained the experiments byt these 6 lines are not the part of this module"""
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.manplt43 =conv_2nV1(128,320,128)
        self.manplt32 =conv_2nV1(64,128,64)
        #self.manplt21 =conv_2nV1(64,128,64)
        self.manplt21 =SqueezeExcitation(64,64)
      

    def forward(self, x4, x3, x2,x1):
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        #print('x3',x3.shape)
        #print('x4',x4.shape)
        x43 =self.manplt43(x3,x4)
        #print("x43",x43.shape)
        x32 =self.manplt32(x2,x43)
        #print("x32",x32.shape)
        x =x1+x32 #x =x1+self.upsample(x32) (orignal model code but for this test it is not upsampled)
        x21 =self.manplt21(x)
        #print("x21",x21.shape)
        #out = self.upsample(x21)
        return x21

class ACFME(nn.Module):
    def __init__(self, in_plans,out_plans):
        super(ACFME, self).__init__()

        self.cha = ChannelAttention(in_planes=in_plans)
        self.spa = SpatialAttention()
        self.conv = BasicConv2d(in_plans, out_plans, kernel_size=3, stride=1, padding=1, relu=True)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        
        cha = self.cha(x)
        spa =self.spa(cha)
        xy = cha+spa
        xo = x * xy 
        xo = self.upsample(self.conv(xo))

        return xo    
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) 
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
#######################


class PolypPVTUpdated(nn.Module):
    def __init__(self, channel=64):
        super(PolypPVTUpdated, self).__init__()

        ### loading the VIT model
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
      
        ### Applying RFB module for re-channel  
          
        self.acfm43 = ACFME(512,320)
        self.acfm32 = ACFME(320,128)
        self.acfm21 = ACFME(128,64)
        
        ### SA module
        
        self.chata =ChannelAttention(64)    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(channel, 1, 1)

        #self.rfb1 = nn.Conv2d(64,channel,1)   
        # self.rf2 = nn.Conv2d(64,128,1)  
        # self.rf3 = nn.Conv2d(128,320,1)  
        # self.rf4 = nn.Conv2d(320,512,1) 
        
        ### cascade Feature fusion module
        self.CFM = CFM()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print("x1",x1.shape)
        # print("x2",x2.shape)
        # print("x3",x3.shape)
        # print("x4",x4.shape)
        # ACFM Module
        acfm34 = self.acfm43(x4)
        #acfm34  = self.upsample(acfm34)
        acfm34a =acfm34 +x3
        #print('acfm4',acfm34.shape)
        #print('rfbx2',rfb2.shape)
        acfm23 = self.acfm32(acfm34a)
        #acfm23  = self.upsample(acfm23)
        acfm23a =acfm23 +x2
        #print('acfm3',acfm23.shape)
        #print('rfbx1',rfb1.shape)
        acfm12= self.acfm21(acfm23a)
        chata = self.chata(x1)
        #print('acfm2',acfm12.shape)
        ## can remove it
        #acfm12out  = self.upsample4(acfm12)
        ## can remove it
        # acfm34  = self.upsample4(acfm34)
        ### CFM(left, down)
        # cfm_feature = self.CFM(acfm34,acfm23,acfm12)
        #out = chata+acfm12

        # rf4 = self.rf4(acfm34)
        # rf3 = self.rf3(acfm23)
        # rf2 = self.rf2(acfm12)
        out = chata+acfm12
        out = self.upsample(out)
        #print("out",out.shape)
        ### CFM(left, down)
        cfm_feature = self.CFM(acfm34,acfm23,acfm12,out)

        prediction1 = self.out_CFM(cfm_feature)
        #print('prediction1',prediction1 .shape)
        prediction1_8 = F.interpolate(prediction1, scale_factor=2, mode='bilinear') 
        #print('prediction1_8 ',prediction1_8 .shape)
        return prediction1_8
class PolypPVT(nn.Module):
    def __init__(self, channel=64):
        super(PolypPVT, self).__init__()

        ### loading the VIT model
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
      
        ### Applying RFB module for re-channel  
        self.rfb1 = RFB_modified(64,channel)   
        self.rfb2 = RFB_modified(128,channel)  
        self.rfb3 = RFB_modified(320,channel)  
        self.rfb4 = RFB_modified(512,channel)   
        
        ### ACFM Module
        self.acfm43 = ACFME()
        self.acfm32 = ACFME()
        self.acfm21 = ACFME()
        
        ### SA module
        #self.sa = DGCM()
        ### cascade Feature fusion module
        self.CFM = CFM(channel)
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        # print('x4',x4.shape)
        ### RFB modules
        rfb1,rfb2,rfb3,rfb4 = self.rfb1(x1),self.rfb2(x2),self.rfb3(x3),self.rfb4(x4)
        # print('rfbx1',rfb1.shape)
        # print('rfbx2',rfb2.shape)
        # print('rfbx3',rfb3.shape)
        # print('rfbx4',rfb4.shape)
        #rfb1 = self.sa(rfb1)
        #rfb1 = self.upsample(rfb1)
        rfb2 = self.upsample(rfb2)
        rfb3 = self.upsample(rfb3)
        rfb4 = self.upsample(rfb4)
        # print('rfbx2up',rfb2.shape)
        # print('rfbx3up',rfb3.shape)
        # print('rfbx4up',rfb4.shape)
        # ACFM Module
        acfm34 = self.acfm43(rfb3,rfb4)
        #acfm34  = self.upsample(acfm34)
        #print('acfm4',acfm34.shape)
        #print('rfbx2',rfb2.shape)
        acfm23 = self.acfm32(rfb2,acfm34)
        acfm23  = self.upsample(acfm23)
        #print('acfm3',acfm23.shape)
        #print('rfbx1',rfb1.shape)
        acfm12= self.acfm21(acfm23,rfb1)
        
        #print('acfm2',acfm12.shape)
        ## can remove it
        acfm34  = self.upsample4(acfm34)
        ### CFM(left, down)
        cfm_feature = self.CFM(acfm34,acfm23,acfm12)

        prediction1 = self.out_CFM(cfm_feature)
        #print('prediction1',prediction1 .shape)
        prediction1_8 = F.interpolate(prediction1, scale_factor=2, mode='bilinear') 
        #print('prediction1_8 ',prediction1_8 .shape)
        return prediction1_8
class PolypPVTRFB(nn.Module):
    def __init__(self, channel=64):
        super(PolypPVTRFB, self).__init__()

        ### loading the VIT model
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
      
        ### Applying RFB module for re-channel 
        #self.chat = ChannelAttention(64) 
        self.rfb1 = RFB_modified(64,channel)   
        self.rfb2 = RFB_modified(128,64)  
        self.rfb3 = RFB_modified(320,128)  
        self.rfb4 = RFB_modified(512,320)   
        
        ### ACFM Module
        # self.acfm43 = ACFM()
        # self.acfm32 = ACFM()
        # self.acfm21 = ACFM()
        
        ### SA module
        #self.sa = DGCM()
        ### cascade Feature fusion module
        # self.CFM = CFM(channel)
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        # print('x4',x4.shape)
        ### RFB modules
        rfb4 = self.upsample(self.rfb4(x4))
        rfb4 = rfb4+x3
        rfb3=self.upsample(self.rfb3(rfb4))
        rfb3 = rfb3+x2
        rfb2=self.upsample(self.rfb2(rfb3))
        rfb2 = rfb2+x1
        rfb1= self.upsample(self.rfb1(rfb2))
        # print('rfbx1',rfb1.shape)
        # print('rfbx2',rfb2.shape)
        # print('rfbx3',rfb3.shape)
        # print('rfbx4',rfb4.shape)
        #rfb1 = self.sa(rfb1)
        #rfb1 = self.upsample(rfb1)
        #rfb2 = self.upsample(rfb2)
        #rfb3 = self.upsample(rfb3)
        
        # print('rfbx2up',rfb1.shape)
        # print('rfbx2up',rfb2.shape)
        # print('rfbx3up',rfb3.shape)
        # print('rfbx4up',rfb4.shape)
        # ACFM Module
        # acfm34 = self.acfm43(rfb3,rfb4)
        #acfm34  = self.upsample(acfm34)
        #print('acfm4',acfm34.shape)
        #print('rfbx2',rfb2.shape)
        # acfm23 = self.acfm32(rfb2,acfm34)
        # acfm23 =rfb2+acfm34
        # acfm23  = self.upsample(acfm23)
        #print('acfm3',acfm23.shape)
        #print('rfbx1',rfb1.shape)
        # acfm12= self.acfm21(acfm23,rfb1)
        
        #print('acfm2',acfm12.shape)
        ## can remove it
        # acfm34  = self.upsample4(acfm34)
        ### CFM(left, down)
        # cfm_feature = self.CFM(acfm34,acfm23,acfm12)

        prediction1 = self.out(rfb1)
        #print('prediction1',prediction1 .shape)
        prediction1_8 = F.interpolate(prediction1 , scale_factor=2, mode='bilinear') 
        #print('prediction1_8 ',prediction1_8 .shape)
        return prediction1_8
class PolypPVTACFM(nn.Module):
    def __init__(self, channel=64):
        super(PolypPVTACFM, self).__init__()

        ### loading the VIT model
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
      
        ### Applying RFB module for re-channel  
        # self.rfb1 = RFB_modified(64,channel)   
        # self.rfb2 = RFB_modified(128,64)  
        # self.rfb3 = RFB_modified(320,128)  
        # self.rfb4 = RFB_modified(512,320)   
        # self.rfb1 = nn.Conv2d(64,channel,1)   
        # self.rfb2 = nn.Conv2d(128,channel,1)  
        # self.rfb3 = nn.Conv2d(320,channel,1)  
        # self.rfb4 = nn.Conv2d(512,channel,1)   
        ### ACFM Module
        self.acfm43 = ACFME(512,320)
        self.acfm32 = ACFME(320,128)
        self.acfm21 = ACFME(128,64)
        
        ### SA module
        #self.sa = DGCM()
        ### cascade Feature fusion module
        # self.CFM = CFM(channel)
        self.chata =ChannelAttention(64)    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        # print('x4',x4.shape)
        ### RFB modules
        # rfb4 = self.upsample(self.rfb4(x4))
        # rfb4 = rfb4+x3
        # rfb3=self.upsample(self.rfb3(rfb4))
        # rfb3 = rfb3+x2
        # rfb2=self.upsample(self.rfb2(rfb3))
        # rfb2 = rfb2+x1
        # rfb1= self.upsample(self.rfb1(rfb2))
        # print('rfbx1',rfb1.shape)
        # print('rfbx2',rfb2.shape)
        # print('rfbx3',rfb3.shape)
        # print('rfbx4',rfb4.shape)
        #rfb1 = self.sa(rfb1)
        #rfb1 = self.upsample(rfb1)
        
        # ACFM Modul
        # rfb1 = self.rfb1(x1)
        # rfb2 = self.rfb2(x2)
        # rfb3 = self.rfb3(x3)
        # rfb4 = self.rfb4(x4)

        #rfb2 = self.upsample(rfb2)
        #rfb3 = self.upsample(rfb3)
        
        # rfb2 = self.upsample(rfb2)
        # rfb3 = self.upsample(rfb3)
        # rfb4 = self.upsample(rfb4)
        # print('rfbx2up',rfb2.shape)
        # print('rfbx3up',rfb3.shape)
        # print('rfbx4up',rfb4.shape)
        # ACFM Module
        acfm34 = self.acfm43(x4)
        #acfm34  = self.upsample(acfm34)
        acfm34a =acfm34 +x3
        #print('acfm4',acfm34.shape)
        #print('rfbx2',rfb2.shape)
        acfm23 = self.acfm32(acfm34a)
        #acfm23  = self.upsample(acfm23)
        acfm23a =acfm23 +x2
        #print('acfm3',acfm23.shape)
        #print('rfbx1',rfb1.shape)
        acfm12= self.acfm21(acfm23a)
        chata = self.chata(x1)
        #print('acfm2',acfm12.shape)
        ## can remove it
        #acfm12out  = self.upsample4(acfm12)
        ## can remove it
        # acfm34  = self.upsample4(acfm34)
        ### CFM(left, down)
        # cfm_feature = self.CFM(acfm34,acfm23,acfm12)
        out = chata+acfm12
        prediction1 = self.upsample(self.out(out))
        #print('prediction1',prediction1 .shape)
        prediction1_8 = F.interpolate(prediction1 , scale_factor=2, mode='bilinear') 
        #print('prediction1_8 ',prediction1_8 .shape)
        return prediction1_8
    
class PolypPVTUNET(nn.Module):
    def __init__(self, channel=64):
        super(PolypPVTUNET, self).__init__()

        ### loading the VIT model
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
      
        ### Applying RFB module for re-channel  
        # self.rfb1 = RFB_modified(64,channel)   
        # self.rfb2 = RFB_modified(128,64)  
        # self.rfb3 = RFB_modified(320,128)  
        # self.rfb4 = RFB_modified(512,320)   
        self.rfb1 = nn.Conv2d(64,channel,1)   
        self.rfb2 = nn.Conv2d(128,64,1)  
        self.rfb3 = nn.Conv2d(320,128,1)  
        self.rfb4 = nn.Conv2d(512,320,1)   
        ### ACFM Module
        # self.acfm43 = ACFM()
        # self.acfm32 = ACFM()
        # self.acfm21 = ACFM()
        
        ### SA module
        #self.sa = DGCM()
        ### cascade Feature fusion module
        # self.CFM = CFM(channel)
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        # print('x4',x4.shape)
        ### RFB modules
        rfb4 = self.upsample(self.rfb4(x4))
        rfb4 = rfb4+x3
        rfb3=self.upsample(self.rfb3(rfb4))
        rfb3 = rfb3+x2
        rfb2=self.upsample(self.rfb2(rfb3))
        rfb2 = rfb2+x1
        rfb1= self.upsample(self.rfb1(rfb2))
        # print('rfbx1',rfb1.shape)
        # print('rfbx2',rfb2.shape)
        # print('rfbx3',rfb3.shape)
        # print('rfbx4',rfb4.shape)
        #rfb1 = self.sa(rfb1)
        #rfb1 = self.upsample(rfb1)
        #rfb2 = self.upsample(rfb2)
        #rfb3 = self.upsample(rfb3)
        
        # print('rfbx2up',rfb1.shape)
        # print('rfbx2up',rfb2.shape)
        # print('rfbx3up',rfb3.shape)
        # print('rfbx4up',rfb4.shape)
        # ACFM Module
        # acfm34 = self.acfm43(rfb3,rfb4)
        #acfm34  = self.upsample(acfm34)
        #print('acfm4',acfm34.shape)
        #print('rfbx2',rfb2.shape)
        # acfm23 = self.acfm32(rfb2,acfm34)
        # acfm23 =rfb2+acfm34
        # acfm23  = self.upsample(acfm23)
        #print('acfm3',acfm23.shape)
        #print('rfbx1',rfb1.shape)
        # acfm12= self.acfm21(acfm23,rfb1)
        
        #print('acfm2',acfm12.shape)
        ## can remove it
        # acfm34  = self.upsample4(acfm34)
        ### CFM(left, down)
        # cfm_feature = self.CFM(acfm34,acfm23,acfm12)

        prediction1 = self.out(rfb1)
        #print('prediction1',prediction1 .shape)
        prediction1_8 = F.interpolate(prediction1 , scale_factor=2, mode='bilinear') 
        #print('prediction1_8 ',prediction1_8 .shape)
        return prediction1_8
class PolypPVTCFMUpdated(nn.Module):
    def __init__(self, channel=64):
        super(PolypPVTCFMUpdated, self).__init__()

        ### loading the VIT model
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
      
        ### Applying RFB module for re-channel  
        # self.rfb1 = RFB_modified(64,channel)   
        self.rfb2 = RFB_modified(128,64)  
        self.rfb3 = RFB_modified(320,128)  
        self.rfb4 = RFB_modified(512,320)   
        # self.rfb1 = nn.Conv2d(64,channel,1)   
        # self.rfb2 = nn.Conv2d(128,channel,1)  
        # self.rfb3 = nn.Conv2d(320,channel,1)  
        # self.rfb4 = nn.Conv2d(512,channel,1)   
        ### ACFM Module
        # self.acfm43 = ACFM()
        # self.acfm32 = ACFM()
        # self.acfm21 = ACFM()
        
        ### SA module
        #self.sa = DGCM()
        ### cascade Feature fusion module
        self.CFM = CFM()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        # print('x4',x4.shape)
        ### RFB modules
        rfb4 = self.upsample(self.rfb4(x4))
        #print('rfbx4',rfb4.shape)
        rfb4 = rfb4+x3
        rfb3=self.upsample(self.rfb3(rfb4))
        #print('rfbx3',rfb3.shape)
        rfb3 = rfb3+x2
        rfb2=self.rfb2(rfb3)
        #print('rfbx2',rfb2.shape)
        rfb2 = self.upsample(rfb2)+x1
        #rfb1= self.upsample(self.rfb1(rfb2))
        # print('rfbx1',rfb1.shape)
        # print('rfbx2',rfb2.shape)
        # print('rfbx3',rfb3.shape)
        # print('rfbx4',rfb4.shape)
        #rfb1 = self.sa(rfb1)
        #rfb1 = self.upsample(rfb1)
        
        # ACFM Modul
        # rfb1 = self.rfb1(x1)
        # rfb2 = self.rfb2(x2)
        # rfb3 = self.rfb3(x3)
        # rfb4 = self.rfb4(x4)

        #rfb2 = self.upsample(rfb2)
        #rfb3 = self.upsample(rfb3)
        
        #rfb2 = self.upsample(rfb2)
        # rfb3 = self.upsample(rfb3)
        # rfb4 = self.upsample4(rfb4)
        # print('rfbx2up',rfb2.shape)
        # print('rfbx3up',rfb3.shape)
        # print('rfbx4up',rfb4.shape)
        # ACFM Module
        # acfm34 = self.acfm43(rfb3,rfb4)
        # #acfm34  = self.upsample(acfm34)
        # #print('acfm4',acfm34.shape)
        # #print('rfbx2',rfb2.shape)
        # acfm23 = self.acfm32(rfb2,acfm34)
        # acfm23  = self.upsample(acfm23)
        # #print('acfm3',acfm23.shape)
        # #print('rfbx1',rfb1.shape)
        # acfm12= self.acfm21(acfm23,rfb1)
        
        # #print('acfm2',acfm12.shape)
        # ## can remove it
        # acfm34  = self.upsample4(acfm34)
        ## can remove it
        # acfm34  = self.upsample4(acfm34)
        ### CFM(left, down)
        cfm_feature = self.CFM(rfb4,rfb3,rfb2,x1)
        #print('cfm featur',cfm_feature.shape)
        prediction1 = self.out(cfm_feature)
        #print('prediction1',prediction1 .shape)
        prediction1_8 = F.interpolate(prediction1 , scale_factor=4, mode='bilinear') 
        #print('prediction1_8 ',prediction1_8 .shape)
        return prediction1_8

"""checking RFB importance in Main model"""

# class PolypPVT(nn.Module):
#     def __init__(self, channel=64):
#         super(PolypPVT, self).__init__()

#         ### loading the VIT model
#         self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
#         path = './pretrained_pth/pvt_v2_b2.pth'
#         save_model = torch.load(path)
#         model_dict = self.backbone.state_dict()
#         state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
#         model_dict.update(state_dict)
#         self.backbone.load_state_dict(model_dict)
      
#         ### Applying RFB module for re-channel  
#         # self.rfb1 = RFB_modified(64,channel)   
#         # self.rfb2 = RFB_modified(128,channel)  
#         # self.rfb3 = RFB_modified(320,channel)  
#         # self.rfb4 = RFB_modified(512,channel)   
#         self.Translayer2_0 = BasicConv2d(64, channel, 1)
#         self.Translayer2_1 = BasicConv2d(128, channel, 1)
#         self.Translayer3_1 = BasicConv2d(320, channel, 1)
#         self.Translayer4_1 = BasicConv2d(512, channel, 1)
#         ### ACFM Module
#         self.acfm43 = ACFM()
#         self.acfm32 = ACFM()
#         self.acfm21 = ACFM()
        
#         ### SA module
#         #self.sa = DGCM()
#         ### cascade Feature fusion module
#         self.CFM = CFM(channel)
                
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.out_CFM = nn.Conv2d(channel, 1, 1)


#     def forward(self, x):

#         # backbone
#         pvt = self.backbone(x)
#         x1 = pvt[0]
#         x2 = pvt[1]
#         x3 = pvt[2]
#         x4 = pvt[3]
#         # print('x1',x1.shape)
#         # print('x2',x2.shape)
#         # print('x3',x3.shape)
#         # print('x4',x4.shape)
#         ### RFB modules
#         rfb1,rfb2,rfb3,rfb4 = self.Translayer2_0(x1),self.Translayer2_1(x2),self.Translayer3_1(x3),self.Translayer4_1(x4)
#         # print('rfbx1',rfb1.shape)
#         # print('rfbx2',rfb2.shape)
#         # print('rfbx3',rfb3.shape)
#         # print('rfbx4',rfb4.shape)
#         #rfb1 = self.sa(rfb1)
#         #rfb1 = self.upsample(rfb1)
#         rfb2 = self.upsample(rfb2)
#         rfb3 = self.upsample(rfb3)
#         rfb4 = self.upsample(rfb4)
#         # print('rfbx2up',rfb2.shape)
#         # print('rfbx3up',rfb3.shape)
#         # print('rfbx4up',rfb4.shape)
#         # ACFM Module
#         acfm34 = self.acfm43(rfb3,rfb4)
#         #acfm34  = self.upsample(acfm34)
#         #print('acfm4',acfm34.shape)
#         #print('rfbx2',rfb2.shape)
#         acfm23 = self.acfm32(rfb2,acfm34)
#         acfm23  = self.upsample(acfm23)
#         #print('acfm3',acfm23.shape)
#         #print('rfbx1',rfb1.shape)
#         acfm12= self.acfm21(acfm23,rfb1)
        
#         #print('acfm2',acfm12.shape)
#         ## can remove it
#         acfm34  = self.upsample4(acfm34)
#         ### CFM(left, down)
#         cfm_feature = self.CFM(acfm34,acfm23,acfm12)

#         prediction1 = self.out_CFM(cfm_feature)
#         #print('prediction1',prediction1 .shape)
#         prediction1_8 = F.interpolate(prediction1, scale_factor=2, mode='bilinear') 
#         #print('prediction1_8 ',prediction1_8 .shape)
#         return prediction1_8

# """Checking ACFM importance"""
# class PolypPVT(nn.Module):
#     def __init__(self, channel=64):
#         super(PolypPVT, self).__init__()

#         ### loading the VIT model
#         self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
#         path = './pretrained_pth/pvt_v2_b2.pth'
#         save_model = torch.load(path)
#         model_dict = self.backbone.state_dict()
#         state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
#         model_dict.update(state_dict)
#         self.backbone.load_state_dict(model_dict)
      
#         ### Applying RFB module for re-channel  
#         self.rfb1 = RFB_modified(64,channel)   
#         self.rfb2 = RFB_modified(128,channel)  
#         self.rfb3 = RFB_modified(320,channel)  
#         self.rfb4 = RFB_modified(512,channel)   
        
#         ### ACFM Module
#         # self.acfm43 = ACFM()
#         # self.acfm32 = ACFM()
#         # self.acfm21 = ACFM()
        
#         ### SA module
#         #self.sa = DGCM()
#         ### cascade Feature fusion module
#         self.CFM = CFM(channel)
                
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.out_CFM = nn.Conv2d(channel, 1, 1)


#     def forward(self, x):

#         # backbone
#         pvt = self.backbone(x)
#         x1 = pvt[0]
#         x2 = pvt[1]
#         x3 = pvt[2]
#         x4 = pvt[3]
#         # print('x1',x1.shape)
#         # print('x2',x2.shape)
#         # print('x3',x3.shape)
#         # print('x4',x4.shape)
#         ### RFB modules
#         rfb1,rfb2,rfb3,rfb4 = self.rfb1(x1),self.rfb2(x2),self.rfb3(x3),self.rfb4(x4)
#         # print('rfbx1',rfb1.shape)
#         # print('rfbx2',rfb2.shape)
#         # print('rfbx3',rfb3.shape)
#         # print('rfbx4',rfb4.shape)
#         #rfb1 = self.sa(rfb1)
#         #rfb1 = self.upsample(rfb1)
#         rfb2 = self.upsample(rfb2)
#         rfb3 = self.upsample(rfb3)
#         rfb3  = self.upsample(rfb3)
#         rfb4 = self.upsample(rfb4)
#         rfb4  = self.upsample4(rfb4)
#         # print('rfbx2up',rfb2.shape)
#         # print('rfbx3up',rfb3.shape)
#         # print('rfbx4up',rfb4.shape)
#         # ACFM Module
#         # acfm34 = self.acfm43(rfb3,rfb4)
#         # #acfm34  = self.upsample(acfm34)
#         # #print('acfm4',acfm34.shape)
#         # #print('rfbx2',rfb2.shape)
#         # acfm23 = self.acfm32(rfb2,acfm34)
#         # acfm23  = self.upsample(acfm23)
#         # #print('acfm3',acfm23.shape)
#         # #print('rfbx1',rfb1.shape)
#         # acfm12= self.acfm21(acfm23,rfb1)
        
#         # #print('acfm2',acfm12.shape)
#         # ## can remove it
#         # acfm34  = self.upsample4(acfm34)
#         ### CFM(left, down)
#         #cfm_feature = self.CFM(acfm34,acfm23,acfm12)
#         cfm_feature = self.CFM(rfb2,rfb3,rfb4)
#         prediction1 = self.out_CFM(cfm_feature)
#         #print('prediction1',prediction1 .shape)
#         prediction1_8 = F.interpolate(prediction1, scale_factor=4, mode='bilinear') 
#         #print('prediction1_8 ',prediction1_8 .shape)
#         return prediction1_8

# """Checking CFM importance"""
# class PolypPVT(nn.Module):
#     def __init__(self, channel=64):
#         super(PolypPVT, self).__init__()

#         ### loading the VIT model
#         self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
#         path = './pretrained_pth/pvt_v2_b2.pth'
#         save_model = torch.load(path)
#         model_dict = self.backbone.state_dict()
#         state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
#         model_dict.update(state_dict)
#         self.backbone.load_state_dict(model_dict)
      
#         ### Applying RFB module for re-channel  
#         self.rfb1 = RFB_modified(64,channel)   
#         self.rfb2 = RFB_modified(128,channel)  
#         self.rfb3 = RFB_modified(320,channel)  
#         self.rfb4 = RFB_modified(512,channel)   
        
#         ### ACFM Module
#         self.acfm43 = ACFM()
#         self.acfm32 = ACFM()
#         self.acfm21 = ACFM()
        
#         ### SA module
#         #self.sa = DGCM()
#         ### cascade Feature fusion module
#         #self.CFM = CFM(channel)
                
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.out_CFM = nn.Conv2d(channel, 1, 1)


#     def forward(self, x):

#         # backbone
#         pvt = self.backbone(x)
#         x1 = pvt[0]
#         x2 = pvt[1]
#         x3 = pvt[2]
#         x4 = pvt[3]
#         # print('x1',x1.shape)
#         # print('x2',x2.shape)
#         # print('x3',x3.shape)
#         # print('x4',x4.shape)
#         ### RFB modules
#         rfb1,rfb2,rfb3,rfb4 = self.rfb1(x1),self.rfb2(x2),self.rfb3(x3),self.rfb4(x4)
#         # print('rfbx1',rfb1.shape)
#         # print('rfbx2',rfb2.shape)
#         # print('rfbx3',rfb3.shape)
#         # print('rfbx4',rfb4.shape)
#         #rfb1 = self.sa(rfb1)
#         #rfb1 = self.upsample(rfb1)
#         rfb2 = self.upsample(rfb2)
#         rfb3 = self.upsample(rfb3)
#         # rfb3  = self.upsample(rfb3)
#         rfb4 = self.upsample(rfb4)
#         # rfb4  = self.upsample4(rfb4)
#         # print('rfbx2up',rfb2.shape)
#         # print('rfbx3up',rfb3.shape)
#         # print('rfbx4up',rfb4.shape)
#         # ACFM Module
#         acfm34 = self.acfm43(rfb3,rfb4)
#         #acfm34  = self.upsample(acfm34)
#         #print('acfm4',acfm34.shape)
#         #print('rfbx2',rfb2.shape)
#         acfm23 = self.acfm32(rfb2,acfm34)
#         acfm23  = self.upsample(acfm23)
#         #print('acfm3',acfm23.shape)
#         #print('rfbx1',rfb1.shape)
#         acfm12= self.acfm21(acfm23,rfb1)
        
#         # #print('acfm2',acfm12.shape)
#         # ## can remove it
#         # acfm34  = self.upsample4(acfm34)
#         ### CFM(left, down)
#         #cfm_feature = self.CFM(acfm34,acfm23,acfm12)
#         #cfm_feature = self.CFM(rfb2,rfb3,rfb4)
#         prediction1 = self.out_CFM(acfm12)
#         #print('prediction1',prediction1 .shape)
#         prediction1_8 = F.interpolate(prediction1, scale_factor=2, mode='bilinear') 
#         #print('prediction1_8 ',prediction1_8 .shape)
#         return prediction1_8
def count_par(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    model = PolypPVT().cuda()
    
    total_par = count_par(model)
    print(total_par)
    #input_tensor = torch.randn(1, 3, 224, 224).cuda()
    
    # #orignal model parameters =25,107,604, new = 25237653
    #summary(model, input_tensor)