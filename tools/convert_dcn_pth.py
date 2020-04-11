import torch

origin_pth = torch.load('./weights/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_converted.pth')
origin_state_dict = origin_pth['state_dict']

new_state_dict = {}
for k,v in origin_state_dict.items():
    if 'neck' in k:
        new_k = k[:5] + '0.' + k[5:]
        new_state_dict[new_k] = v
    else:
        new_state_dict[k] = v

states = {'meta':origin_pth['meta'], 'state_dict':new_state_dict}
torch.save(states, './weights/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_converted2.pth')


# origin_pth = torch.load('./weights/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth')
# origin_state_dict = origin_pth['state_dict']

# new_state_dict = {}
# for k,v in origin_state_dict.items():
#     if 'offset' in k:
#         index = k.find('conv2_offset')
#         new_k = k[:index] + 'conv2.conv_offset.' + k.split('.')[-1]
#         new_state_dict[new_k] = v
#     else:
#         new_state_dict[k] = v

# states = {'meta':origin_pth['meta'], 'state_dict':new_state_dict}
# torch.save(states, './weights/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_converted.pth')

# origin_pth = torch.load('./weights/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_converted.pth')
# origin_state_dict = origin_pth['state_dict']

# new_state_dict = {}
# for k,v in origin_state_dict.items():
#     if 'offset' in k:
#         index = k.find('conv2')
#         new_k = k[:index] + k[index+6:]
#         new_k = new_k.replace('conv_offset', 'conv2_offset')
#         new_state_dict[new_k] = v
#     else:
#         new_state_dict[k] = v

# states = {'meta':origin_pth['meta'], 'state_dict':new_state_dict}
# torch.save(states, './weights/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_converted.pth_20190408-0e50669c.pth')

