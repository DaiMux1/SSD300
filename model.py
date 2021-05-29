from torch._C import device
from lib import *
from l2_norm import L2Norm
from default_box import DefBox

def create_vgg():
    layers = []
    in_channels = 3
    
    cfgs = [64, 64, 'M', 128, 128, 'M',
            256, 256, 256, 'MC', 512, 512, 512, 'M',
            512, 512, 512]
    
    for cfg in cfgs:
        if cfg == 'M': # lam tron xuong
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC': # lam tron len , ceiling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1) #cfg la output
            # nn.conv2d()
            layers += [conv2d, nn.ReLU(inplace=True)] # inplace quyet dinh input co luu lai hay ko, implace=True la k luu
            in_channels = cfg
            # dau ra co so channel=64
            
    pools5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024,1024, kernel_size=1)
    
    layers += [pools5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    
    return nn.ModuleList(layers)

def create_extras():
    layers = []
    in_channels = 1024
    
    cfgs = [256, 512, 128, 256, 128, 256, 128, 256] # dau ra cua layer nay la dau vao cua layer ke tiep
    
    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]
    
    return nn.ModuleList(layers)
    
def create_loc_conf(num_classes=21, bbox_ratio_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []
    
    #source1
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]
    
    #source2
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]
    
    #source3
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size=3, padding=1)]
    
    #source4
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]
    
    #source5
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4]*num_classes, kernel_size=3, padding=1)]
    
    #source6
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5]*num_classes, kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

cfg = {
    'num_classes': 21, #VOC data co 20 classes + 1 background class
    'input_size': 300,
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4], # source 1 -> 6  ty le khung hinh cho cac source
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300], # size of default box
    'min_size': [30, 60, 111, 162, 213, 264],
    'max_size': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]]
    
}

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        
        #tao main modules
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(cfg['num_classes'], cfg['bbox_aspect_num'])
        self.L2Norm = L2Norm()
        
        # tao ra 8732 default box 
        dbox = DefBox(cfg) 
        self.dbox_list = dbox.create_defbox()
        
        if phase == 'inference':
            self.detect = Detect()
        
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        
        for k in range(23):
            x = self.vgg[k](x)
            
        source1 = self.L2Norm(x)
        sources.append(source1)
        
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            
        # source2
        sources.append(x)
        
        #source3->6
        for k, v in  enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            
            if k % 2 == 1:
                sources.append(x)
            
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # aspect_ratio_num =4 or 6
            # (batch_size, 4*aspect_ratio_num, featuremap_height, featuremap_width)
            # ->(batch_size, featuremap_height, featuremap_width, 4*aspect_ratio_num)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for  o in loc], 1) # (batch_size,34928) 4 * 8732
        conf = torch.cat([o.view(o.size(0), -1) for  o in conf], 1) # (batch_size,8732*21) 
        
        loc = loc.view(loc.size(0), -1 ,4) #(batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes) #(batch_size, 8732, 21)
            
        output = (loc, conf, self.dbox_list)
        
        if self.phase == 'inference':
            return self.detect(output[0], output[1], output[2])
        else:
            return output
            
            
        

def decode(loc, defbox_list):
    """
    parameters
        loc: [8732, 4] offset information, location (delta_x, delta_y, delta_w, delta_h)
        defbox_list: [8732, 4] (cx_d, cy_d, w_d, h_d)
        
    returns:
        boxes [xmin, ymin, xmax, ymax]
    """
    
    boxes = torch.cat((defbox_list[:, :2] + loc[:, :2]*defbox_list[:, 2:]*0.1),
                      (defbox_list[:, 2:]*torch.exp(loc[:, 2:]*0.2)), dim=1)
    
    boxes[:, :2] -= boxes[:, 2:]/2 #tinh xmin, ymin
    boxes[:, 2:] += boxes[:, :2]  #tinh xmax, ymax

    return boxes

#non-maximun supression
# scores là độ tự tin
#top_k chi lay bbox co do tu tin cao nhat
def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    boxes:[num_box, 4]
    scores: [num_box]
    
    """
    count = 0
    keep = scores.new(scores.size(0)).zero_().long() # dịnh dạng giống nhau
    
    # boxes toa do
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    
    #dien tich cua boxes
    area = torch.mul(x2-x1, y2-y1)
    
    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()
    
    value, idx = scores.sort(0) # sort theo dim=0
    
    idx = idx[-top_k:] # lay top_k phan tu cuoi
    
    while idx.numel() > 0: # lay het gia tri trong idx
        i = idx[-1] # có độ tự tin lớn nhất

        keep[count] = i
        count += 1
        
        if idx.size(0) == 1: 
            break
        
        idx = idx[:-1] #id cua boxes ngoại trừ box có độ tự tin cao nhất
        
        # lay thong tin cuar boxes
        torch.index_select(x1, 0, ind, out=tmp_x1)  #x1
        torch.index_select(y1, 0, ind, out=tmp_y1)  #y1
        torch.index_select(x2, 0, ind, out=tmp_x2)  #x2
        torch.index_select(y2, 0, ind, out=tmp_y2)  #y2
        
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # = x1[i] if tmp_x1 < x1[1]
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_x2 = torch.clamp(tmp_y2, max=y2[i])# = y2[i] if tmp_y2 > y2[i]
        
        # chuyển về tensor có size mà insex được giảm đi 1
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)
        
        # tim dien tich overlap
        inter = tmp_w * tmp_h
        
        others_area = torch.index_select(area, 0, idx) # lay ra dt cua moi bbox
        union = area[i] + others_area - inter
        
        iou = inter / union
        
        idx = idx[iou.le(overlap)] # nho honn overlap thi giu lai
        
    return keep, count
        
        
class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        
    def forward(self, loc_data, conf_data, dbox_list):
        num_batch = loc_data.size(0) #batch_size (2, 4, 8,...., 64, 128   )
        num_dbox = loc_data.size(1) #8732
        num_classes = conf_data.size(2) #21
   
        conf_data = self.softmax(conf_data) 
        #(batch_num, num_dbox, num_classes) -> (batch_num, num_classes,num_dbox)
        conf_preds = conf_data.transpose(2, 1)
        
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        #xu ly tung anh torn g1 batch cac buc anh
        for i in range(num_batch):
            #tinh bbox từ offset information va default box
            decode_boxes = decode(loc=loc_data[i], defbox_list=dbox_list)
            
            # copy confidence score của ảnh thứ i
            conf_scores = conf_preds[i].clone()
            
            for cl in range(1, num_classes):
                c_mask = conf_preds[cl].gt(self.conf_thresh) #lay lon hon conf_thresh (0.01) tra ra 1 list [0,1] voi 1 la lon hon conf_thresh
                scores = conf_preds[cl][c_mask]
                
                if scores.nelements() == 0: # co the dung numel()
                    continue
                
                # dua chieu về giống chiều của decode_bõ dể tính toans
                l_mask = c_mask.unsquzee(1).expand_as(decode_boxes) #(8732, 4)
                
                boxes = decode_boxes[l_mask].view(-1, 4)
                
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsquzee(1), boxes[ids[:count]]), 1) # theo chieu ngang
                
        return output
                
                

if __name__ == '__main__':
    # vgg = create_vgg()
    # # print(vgg)
    # extras = create_extras()
    # # print(extras)
    # loc, conf = create_loc_conf()
    # print(loc)
    ssd = SSD(phase='test',cfg=cfg)
    print(ssd)
            