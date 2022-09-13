import torch
import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    def __init__(self):
        super(conv, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pool(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 8, 1, bias=False),
                                nn.LeakyReLU(0.2),
                                nn.Conv3d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# class Self_Attention(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim, activation):
#         super(Self_Attention, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#
#         self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
#
#         self.softmax = nn.Softmax(dim=1)  #
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H X D)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height*Depth)
#         """
#         m_batchsize, C, width, height, depth = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height * depth).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * depth)  # B X C x (d*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         print(energy.shape)
#         attention = self.softmax(energy)  # BX (N) X (N)
#         print(attention.shape)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height * depth)  # B X C X N
#         print(proj_value.shape, attention.shape)
#         out = torch.bmm(proj_value, attention)
#         out = out.view(m_batchsize, C, width, height, depth)
#
#         out = self.gamma * out + x
#         return out


class SASA_Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, num_heads=1, image_size=2, inference=False):
        super(SASA_Layer, self).__init__()
        self.kernel_size = min(kernel_size, image_size)  # receptive field shouldn't be larger than input H/W
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dk % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"

        self.k_conv = nn.Conv3d(self.dk, self.dk, kernel_size=1)
        self.q_conv = nn.Conv3d(self.dk, self.dk, kernel_size=1)
        self.v_conv = nn.Conv3d(self.dv, self.dv, kernel_size=1)

        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn(self.dk // 2, self.kernel_size, 1, 1), requires_grad=True)
        self.rel_encoding_w = nn.Parameter(torch.randn(self.dk // 2, 1, self.kernel_size, 1), requires_grad=True)
        self.rel_encoding_d = nn.Parameter(torch.randn(self.dk // 2, 1, 1, self.kernel_size), requires_grad=True)

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        batch_size, _, height, width, depth = x.size()

        # Compute k, q, v
        padded_x = F.pad(x, [(self.kernel_size - 1) // 2, (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
                             (self.kernel_size - 1) // 2, (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
                             (self.kernel_size - 1) // 2, (self.kernel_size - 1) - ((self.kernel_size - 1) // 2)])
        k = self.k_conv(padded_x)
        q = self.q_conv(x)
        v = self.v_conv(padded_x)

        # Unfold patches into [BS, num_heads*depth, horizontal_patches, vertical_patches, kernel_size, kernel_size]
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
        # Reshape into [BS, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        k = k.reshape(batch_size, self.num_heads, height, width, depth, self.dkh, -1)
        v = v.reshape(batch_size, self.num_heads, height, width, depth, self.dvh, -1)

        # Reshape into [BS, num_heads, height, width, depth_per_head, 1]
        q = q.reshape(batch_size, self.num_heads, height, width, depth, self.dkh, 1)
        qk = torch.matmul(q.transpose(5, 6), k)
        qk = qk.reshape(batch_size, self.num_heads, height, width, depth, self.kernel_size, self.kernel_size,
                        self.kernel_size)
        # Add positional encoding
        qr_h = torch.einsum('bhxytdz,cijk->bhxytijk', q, self.rel_encoding_h)
        qr_w = torch.einsum('bhxytdz,cijk->bhxytijk', q, self.rel_encoding_w)
        qr_d = torch.einsum('bhxytdz,cijk->bhxytijk', q, self.rel_encoding_d)
        qk += qr_h
        qk += qr_w
        qk += qr_d

        qk = qk.reshape(batch_size, self.num_heads, height, width, depth, 1,
                        self.kernel_size * self.kernel_size * self.kernel_size)
        weights = F.softmax(qk, dim=-1)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn_out = torch.matmul(weights, v.transpose(5, 6))
        attn_out = attn_out.reshape(batch_size, -1, height, width, depth)
        return attn_out


from torchinfo import summary


# model = multimodal_Res_att()
# model = multimodal_Res()
# model = Self_Attention(8, nn.LeakyReLU(0.2))
# # model = AttentionBlock(8, 8)
# model.cuda()
# print(torch.cuda.memory_allocated() / 1024 / 1024)
# summary(model, (4, 8, 40, 48, 40))


class Res_Att_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_Att_block, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_2 = nn.BatchNorm3d(out_ch)
        self.relu_2 = nn.LeakyReLU(0.2)
        # self.attention = Self_Attention(out_ch, nn.LeakyReLU(0.2))
        # self.attention = AttentionConv(out_ch, out_ch, 1)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x1, x2):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)

        x2 = self.ca(x2) * x2
        x2 = self.sa(x2) * x2

        x = torch.cat((x, x2), dim=1)
        return x


class Res_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_block, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_2 = nn.BatchNorm3d(out_ch)
        self.relu_2 = nn.LeakyReLU(0.2)

        # self.conv_1_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        # self.batch_1_1 = nn.BatchNorm3d(out_ch)
        # self.attention = Self_Attention(out_ch, nn.ReLU())
        # self.pool = nn.MaxPool3d(2,2)

    def forward(self, x1, x2):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)
        x = torch.cat((x, x2), dim=1)
        # x2 = self.conv_1_1(x2)
        # x2 = self.batch_1_1(x2)
        # x += x2
        return x


class Res_self_Att_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_self_Att_block, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_2 = nn.BatchNorm3d(out_ch)
        self.relu_2 = nn.LeakyReLU(0.2)

        self.attention = SASA_Layer(in_ch)

    def forward(self, x1, x2):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)
        x = self.attention(x)
        x2 = self.attention(x2)
        x = torch.cat((x, x2), dim=1)
        return x


class Mixing_block(nn.Module):
    def __init__(self, in_ch):
        super(Mixing_block, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(in_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1_orig = x1.clone()
        x2_orig = x2.clone()

        x1 = self.conv_1(x1)
        x1 = self.batch_1(x1)
        x1 = self.relu_1(x1)
        x1 = x1_orig + x1

        x2 = self.conv_1(x2)
        x2 = self.batch_1(x2)
        x2 = self.relu_1(x2)
        x2 = x2_orig + x2

        x = torch.cat((x1, x2), dim=1)
        return x


class Mixing_self_att_block(nn.Module):
    def __init__(self, in_ch):
        super(Mixing_self_att_block, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(in_ch)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.attention = SASA_Layer(in_ch)

    def forward(self, x1, x2):
        x1_orig = x1.clone()
        x2_orig = x2.clone()

        x1 = self.conv_1(x1)
        x1 = self.batch_1(x1)
        x1 = self.relu_1(x1)
        x1 = x1_orig + self.attention(x1)

        x2 = self.conv_1(x2)
        x2 = self.batch_1(x2)
        x2 = self.relu_1(x2)
        x2 = x2_orig + self.attention(x2)

        x = torch.cat((x1, x2), dim=1)
        return x


class unimodal_Res(nn.Module):
    def __init__(self):
        super(unimodal_Res, self).__init__()
        self.conv = conv()
        self.pool_1 = nn.MaxPool3d(2, 2)
        self.pool_2 = nn.MaxPool3d(2, 2)
        self.pool_3 = nn.MaxPool3d(2, 2)
        self.pool_4 = nn.MaxPool3d(2, 2)

        self.res_block_1 = Res_block(8, 8)
        self.res_block_2 = Res_block(16, 16)
        self.res_block_3 = Res_block(32, 32)
        self.res_block_4 = Res_block(64, 64)

        ## ADD block ##
        # self.res_block_5 = Res_block(128, 128)
        # self.pool_5 = nn.MaxPool3d(2, 2)
        ## ADD block ##

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 5 * 6 * 5, 1300),
                                        nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block_1(x, x)
        x = self.pool_1(x)

        x = self.res_block_2(x, x)
        x = self.pool_2(x)

        x = self.res_block_3(x, x)
        x = self.pool_3(x)
        x = self.res_block_4(x, x)
        x = self.pool_4(x)
        ## ADD block ##
        # x = self.res_block_5(x, x)
        # x = self.pool_5(x)
        ## ADD block ##
        return self.classifier(x)


# from torchinfo import summary
#
# model = unimodal_Res()
# summary(model, (4, 1, 160, 192, 160))


class multimodal_Res(nn.Module):
    def __init__(self):
        super(multimodal_Res, self).__init__()
        self.conv_1 = conv()
        self.conv_2 = conv()
        self.pool_1 = nn.MaxPool3d(2, 2)

        self.res_block_1_1 = Res_block(8, 8)
        self.res_block_2_1 = Res_block(16, 16)
        self.res_block_3_1 = Res_block(32, 32)
        self.res_block_4_1 = Res_block(64, 64)

        self.res_block_1_2 = Res_block(8, 8)
        self.res_block_2_2 = Res_block(16, 16)
        self.res_block_3_2 = Res_block(32, 32)
        self.res_block_4_2 = Res_block(64, 64)

        ## ADD BLOCK ##
        # self.res_block_5_1 = Res_block(128, 128)
        # self.res_block_5_2 = Res_block(128, 128)
        ## ADD BLOCK ##

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 5 * 6 * 5 * 2, 1300),
                                        nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)
        # print(f'x1 shape : {x1.shape}')
        # print(f'x2 shape : {x2.shape}')
        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x1_orig = x1.clone()
        x2_orig = x2.clone()
        # print(f'x1 shape : {x1.shape}')
        # print(f'x2 shape : {x2.shape}')
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        # print(f'x1 shape : {x1.shape}')
        # print(f'x2 shape : {x2.shape}')
        ## ADD BLOCK ##
        # x1_orig = x1.clone()
        # x2_orig = x2.clone()
        # x1 = self.res_block_5_1(x1_orig, x2_orig)
        # x2 = self.res_block_5_2(x2_orig, x1_orig)
        # x1 = self.pool(x1)
        # x2 = self.pool(x2)
        ## ADD BLOCK ##
        x = torch.cat((x1, x2), dim=1)

        # print(f'x shape : {x.shape}')
        return self.classifier(x)


class mixing_res(nn.Module):
    def __init__(self):
        super(mixing_res, self).__init__()
        self.conv_1 = conv()
        self.conv_2 = conv()
        self.pool_1 = nn.MaxPool3d(2, 2)

        self.res_block_1_1 = Res_block(8, 8)
        self.res_block_2_1 = Res_block(16, 16)
        self.res_block_3_1 = Res_block(32, 32)
        self.res_block_4_1 = Res_block(64, 64)

        self.res_block_1_2 = Res_block(8, 8)
        self.res_block_2_2 = Res_block(16, 16)
        self.res_block_3_2 = Res_block(32, 32)
        self.res_block_4_2 = Res_block(64, 64)

        self.mixing_block = Mixing_block(128)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 5 * 6 * 5 * 2, 1300),
                                        nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))
        # self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 5 * 6 * 5 * 2, 1300),
        #                                 nn.LeakyReLU(0.2), nn.Linear(1300, 500),
        #                                 nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)
        # print(f'x1 shape : {x1.shape}')
        # print(f'x2 shape : {x2.shape}')
        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x1_orig = x1.clone()
        x2_orig = x2.clone()
        # print(f'x1 shape : {x1.shape}')
        # print(f'x2 shape : {x2.shape}')
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        # print(f'x1 shape : {x1.shape}')
        # print(f'x2 shape : {x2.shape}')
        # x = torch.cat((x1, x2), dim=1)
        x = self.mixing_block(x1, x2)
        # print(f'x shape : {x.shape}')
        return self.classifier(x)


class Mixing_block_att(nn.Module):
    def __init__(self, in_ch):
        super(Mixing_block_att, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(in_ch)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()

    def forward(self, x1, x2):
        x1_orig = x1.clone()
        x2_orig = x2.clone()

        x1 = self.conv_1(x1)
        x1 = self.batch_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x1 = x1 + x1_orig

        x2 = self.conv_1(x2)
        x2 = self.batch_1(x2)
        x2 = self.relu_1(x2)
        x2 = self.ca(x2) * x2
        x2 = self.sa(x2) * x2
        x2 = x2 + x2_orig

        x = torch.cat((x1, x2), dim=1)
        return x


class cbr(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cbr, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        return self.relu(x)


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1, device='cpu'):
        super(SpatialSoftmax, self).__init__()

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature).to(device)
        else:
            self.temperature = 1.

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1, feature.shape[1] * feature.shape[2])
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)

        return softmax_attention


class AMGB_model(nn.Module):
    def __init__(self):
        super(AMGB_model, self).__init__()
        self.conv_1 = conv()
        self.conv_2 = conv()

        self.pool = nn.MaxPool3d(2, 2)

        self.res_block_1_1 = Res_block(8, 8)
        self.res_block_2_1 = Res_block(16, 16)
        self.res_block_3_1 = Res_block(32, 32)
        self.res_block_4_1 = Res_block(64, 64)

        self.res_block_1_2 = Res_block(8, 8)
        self.res_block_2_2 = Res_block(16, 16)
        self.res_block_3_2 = Res_block(32, 32)
        self.res_block_4_2 = Res_block(64, 64)

        self.l2_loss = nn.MSELoss(reduction='mean')
        self.scaled_loss = 0.5

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 5 * 6 * 5 * 2, 1300),
                                        nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

    def forward(self, x1, x2):
        x1 = self.conv_1(x1)
        x2 = self.conv_2(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        at_gen_l2_loss = self.at_gen_loss(x1, x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        at_gen_l2_loss += self.at_gen_loss(x1, x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        at_gen_l2_loss += self.at_gen_loss(x1, x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        at_gen_l2_loss += self.at_gen_loss(x1, x2)

        x = torch.cat((x1, x2), dim=1)

        at_gen_l2_loss = self.scaled_loss * at_gen_l2_loss

        return self.classifier(x), at_gen_l2_loss

    def at_gen_loss(self, x1, x2):
        """
        x1 - previous encoder step feature map
        x2 - current encoder step feature map
        """

        # G^2_sum
        sps = SpatialSoftmax(device=x1.device)

        if x1.size() != x2.size():
            x1 = x1.pow(2).sum(dim=1)
            x1 = sps(x1)
            x2 = x2.pow(2).sum(dim=1, keepdim=True)
            x2 = torch.squeeze(self.at_gen_upsample(x2), dim=1)
            x2 = sps(x2)
        else:
            x1 = x1.pow(2).sum(dim=1)
            x1 = sps(x1)
            x2 = x2.pow(2).sum(dim=1)
            x2 = sps(x2)

        loss = self.l2_loss(x1, x2)
        return loss


from torchinfo import summary


# model = mixing_res_self()
# model = multimodal_Res_self()
# model = AMGB_model()
# summary(model, ((4, 1, 160, 192, 160), (4, 1, 160, 192, 160)))

class Res_block_Three(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_block_Three, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.BatchNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_2 = nn.BatchNorm3d(out_ch)
        self.relu_2 = nn.LeakyReLU(0.2)
        # self.attention = Self_Attention(out_ch, nn.ReLU())
        # self.pool = nn.MaxPool3d(2,2)

        self.conv_2_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_2_2 = nn.BatchNorm3d(out_ch)
        self.conv_3_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_3_2 = nn.BatchNorm3d(out_ch)

    def forward(self, x1, x2, x3):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)
        x = torch.cat((x, x2, x3), dim=1)
        # x = x + self.batch_2_2(self.conv_2_1(x2)) + self.batch_3_2(self.conv_3_1(x2))
        return x


class multimodal_three(nn.Module):
    def __init__(self):
        super(multimodal_three, self).__init__()
        self.conv_1 = conv()
        self.conv_2 = conv()
        self.conv_3 = conv()
        self.pool_1 = nn.MaxPool3d(2, 2)

        self.res_block_1_1 = Res_block_Three(8, 8)
        self.res_block_2_1 = Res_block_Three(24, 24)
        self.res_block_3_1 = Res_block_Three(72, 72)
        self.res_block_4_1 = Res_block_Three(216, 216)

        self.res_block_1_2 = Res_block_Three(8, 8)
        self.res_block_2_2 = Res_block_Three(24, 24)
        self.res_block_3_2 = Res_block_Three(72, 72)
        self.res_block_4_2 = Res_block_Three(216, 216)

        self.res_block_1_3 = Res_block_Three(8, 8)
        self.res_block_2_3 = Res_block_Three(24, 24)
        self.res_block_3_3 = Res_block_Three(72, 72)
        self.res_block_4_3 = Res_block_Three(216, 216)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5 * 3, 1300),
                                        nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x1, x2, x3):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)
        x3_orig = self.conv_3(x3)

        x1 = self.res_block_1_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_1_3(x3_orig, x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_2_3(x3_orig, x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_3_3(x3_orig, x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_4_3(x3_orig, x1_orig, x2_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        return self.classifier(x)


class conv3x3_4(nn.Module):
    def __init__(self):
        super(conv3x3_4, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(2)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm3d(2)
        self.relu3 = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return self.pool(x)


class Res_block_instance(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_block_instance, self).__init__()
        self.conv_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batch_1 = nn.InstanceNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.concat_conv = nn.Sequential(*[nn.Conv3d()])

        self.conv_2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_2 = nn.InstanceNorm3d(out_ch)
        self.relu_2 = nn.LeakyReLU(0.2)

        # self.conv_1_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        # self.batch_1_1 = nn.BatchNorm3d(out_ch)
        # self.attention = Self_Attention(out_ch, nn.ReLU())
        # self.pool = nn.MaxPool3d(2,2)

    def forward(self, x1, x2):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)
        x = torch.cat((x, x2), dim=1)
        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Bottlenet_block_instance(nn.Module):
    def __init__(self, in_ch, out_ch, expansion):
        super(Bottlenet_block_instance, self).__init__()
        self.conv_1 = conv1x1x1(in_ch, out_ch)
        self.batch_1 = nn.InstanceNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = conv3x3x3(out_ch, out_ch)
        self.batch_2 = nn.InstanceNorm3d(out_ch)
        self.relu_3 = nn.LeakyReLU(0.2)

        self.conv_3 = conv1x1x1(out_ch, out_ch * expansion)
        self.batch_3 = nn.InstanceNorm3d(out_ch * expansion)

        self.out_relu = nn.LeakyReLU(0.2)
        self.apply_init()

    def forward(self, x1, x2):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_3(x)

        x = self.conv_3(x)
        x = self.batch_3(x)
        x = torch.cat((x, x2), dim=1)

        return self.out_relu(x)

    def apply_init(self):
        torch.nn.init.xavier_normal_(self.conv_1.weight)
        torch.nn.init.xavier_normal_(self.conv_2.weight)
        torch.nn.init.xavier_normal_(self.conv_3.weight)


class Bottlenet_block_instance_three(nn.Module):
    def __init__(self, in_ch, out_ch, expansion):
        super(Bottlenet_block_instance_three, self).__init__()
        self.conv_1 = conv1x1x1(in_ch, out_ch)
        self.batch_1 = nn.InstanceNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = conv3x3x3(out_ch, out_ch)
        self.batch_2 = nn.InstanceNorm3d(out_ch)
        self.relu_3 = nn.LeakyReLU(0.2)

        self.conv_3 = conv1x1x1(out_ch, out_ch * expansion)
        self.batch_3 = nn.InstanceNorm3d(out_ch * expansion)

        self.out_relu = nn.LeakyReLU(0.2)
        self.apply_init()

    def forward(self, x1, x2, x3):
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_3(x)

        x = self.conv_3(x)
        x = self.batch_3(x)
        x = torch.cat((x, x2, x3), dim=1)

        return self.out_relu(x)

    def apply_init(self):
        torch.nn.init.xavier_normal_(self.conv_1.weight)
        torch.nn.init.xavier_normal_(self.conv_2.weight)
        torch.nn.init.xavier_normal_(self.conv_3.weight)


class Bottlenet_block_instance_multimodal(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Bottlenet_block_instance_multimodal, self).__init__()
        expansion = 2
        self.conv_1 = conv1x1x1(in_ch, out_ch)
        # self.batch_1 = nn.InstanceNorm3d(out_ch)
        self.batch_1 = nn.BatchNorm3d(out_ch)
        self.relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = conv3x3x3(out_ch, out_ch)
        # self.batch_2 = nn.InstanceNorm3d(out_ch)
        self.batch_2 = nn.BatchNorm3d(out_ch)
        self.relu_3 = nn.LeakyReLU(0.2)

        self.conv_3 = conv1x1x1(out_ch, out_ch * expansion)
        # self.batch_3 = nn.InstanceNorm3d(out_ch * expansion)
        self.batch_3 = nn.BatchNorm3d(out_ch * expansion)

        self.out_relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        # x1_orig = x1.clone()
        x = self.conv_1(x1)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_3(x)

        x = self.conv_3(x)
        x = self.batch_3(x)
        # x = torch.cat((x, x1_orig, x2), dim=1)
        x = torch.cat((x, x2), dim=1)

        return self.out_relu(x)


class wide_multimodal_res(nn.Module):
    def __init__(self):
        super(wide_multimodal_res, self).__init__()
        self.conv_1 = conv3x3_4()
        self.conv_2 = conv3x3_4()
        self.pool_1 = nn.MaxPool3d(2, 2)
        expansion = 2
        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch
        self.res_block_2_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch
        self.res_block_3_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch
        self.res_block_4_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_1_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch  # 40
        self.res_block_2_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch  # 200
        self.res_block_3_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch  # 1000
        self.res_block_4_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        ## ADD BLOCK ##
        # self.res_block_5_1 = Res_block(128, 128)
        # self.res_block_5_2 = Res_block(128, 128)
        ## ADD BLOCK ##
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5 * 2, 1300),
                                        nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))
        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(625 * 2, 500),
        #                                 nn.LeakyReLU(0.2), nn.Linear(500, 100),
        #                                 nn.LeakyReLU(0.2), nn.Linear(100, 2))

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)

        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.relu(x)

        return self.classifier(x)


class wide_mixing_res(nn.Module):
    def __init__(self):
        super(wide_mixing_res, self).__init__()
        self.conv_1 = conv3x3_4()
        self.conv_2 = conv3x3_4()
        self.pool_1 = nn.MaxPool3d(2, 2)
        expansion = 2
        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch
        self.res_block_2_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch
        self.res_block_3_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch
        self.res_block_4_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_1_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch  # 5
        self.res_block_2_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch  # 15
        self.res_block_3_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 2 + output_ch  # 75
        self.res_block_4_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        self.mixing_block = Mixing_block(648)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5 * 2, 1300),
                                        nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(625 * 2, 500),
        #                                 nn.LeakyReLU(0.2), nn.Linear(500, 100),
        #                                 nn.LeakyReLU(0.2), nn.Linear(100, 2))

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)

        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x = self.mixing_block(x1, x2)
        x = self.relu(x)

        return self.classifier(x)


class residual_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(residual_conv, self).__init__()
        # depth speata
        self.conv_1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.conv_2 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        # self.norm = nn.InstanceNorm3d(out_ch)
        # self.relu = nn.LeakyReLU(0.2)
        self.apply_init()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

    def apply_init(self):
        torch.nn.init.xavier_normal_(self.conv_1.weight)
        torch.nn.init.xavier_normal_(self.conv_2.weight)


class residual_first_wide_multi(nn.Module):
    def __init__(self):
        super(residual_first_wide_multi, self).__init__()
        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_2 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.pool_1 = nn.MaxPool3d(2, 2)
        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance_multimodal(output_ch, output_ch)
        output_ch = output_ch * 2 + output_ch + output_ch
        self.res_block_2_1 = Bottlenet_block_instance_multimodal(output_ch, output_ch)
        output_ch = output_ch * 2 + output_ch + output_ch
        self.res_block_3_1 = Bottlenet_block_instance_multimodal(output_ch, output_ch)
        output_ch = output_ch * 2 + output_ch + output_ch
        self.res_block_4_1 = Bottlenet_block_instance_multimodal(output_ch, output_ch)

        output_ch = 8
        self.res_block_1_2 = Bottlenet_block_instance_multimodal(output_ch, output_ch)
        output_ch = output_ch * 2 + output_ch + output_ch
        self.res_block_2_2 = Bottlenet_block_instance_multimodal(output_ch, output_ch)
        output_ch = output_ch * 2 + output_ch + output_ch
        self.res_block_3_2 = Bottlenet_block_instance_multimodal(output_ch, output_ch)
        output_ch = output_ch * 2 + output_ch + output_ch
        self.res_block_4_2 = Bottlenet_block_instance_multimodal(output_ch, output_ch)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(2048 * 5 * 6 * 5 * 2, 1300),
                                        nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))
        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(2048 * 2, 1300),
        #                                 nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
        #                                 nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)

        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x = torch.cat((x1, x2), dim=1)

        return self.classifier(x)


class residual_first_wide_unimodal(nn.Module):
    def __init__(self, flatten_size):
        super(residual_first_wide_unimodal, self).__init__()
        expansion = 2
        output_ch = 8

        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])

        self.pool = nn.MaxPool3d(2, 2)
        expansion = 2

        self.res_block_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * expansion + output_ch
        self.res_block_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * expansion + output_ch
        self.res_block_3 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * expansion + output_ch
        self.res_block_4 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                        nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))

    def forward(self, x):
        x = self.conv_1(x)

        x = self.res_block_1(x, x)
        x = self.pool(x)

        x = self.res_block_2(x, x)
        x = self.pool(x)

        x = self.res_block_3(x, x)
        x = self.pool(x)

        x = self.res_block_4(x, x)
        x = self.pool(x)

        return self.classifier(x)


class shared_multimodality(nn.Module):
    def __init__(self):
        super(shared_multimodality, self).__init__()
        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_2 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.pool_1 = nn.MaxPool3d(2, 2)
        expansion = 2
        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 3
        self.res_block_2_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 3
        self.res_block_3_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 3
        self.res_block_4_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_1_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 3
        self.res_block_2_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 3
        self.res_block_3_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 3
        self.res_block_4_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1300 * 2, 500), nn.BatchNorm1d(500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.class_1 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))
        self.class_2 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))

        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(2048 * 2, 1300),
        #                                 nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
        #                                 nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))
        # self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5 * 2, 1300),
        #                                 nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
        #                                 nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.shared_conv_1 = nn.Sequential(residual_conv(24, 24), residual_conv(24, 24), nn.BatchNorm3d(24),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_2 = nn.Sequential(residual_conv(72, 72), residual_conv(72, 72), nn.BatchNorm3d(72),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_3 = nn.Sequential(residual_conv(216, 216), residual_conv(216, 216), nn.BatchNorm3d(216),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_4 = nn.Sequential(residual_conv(648, 648), residual_conv(648, 648), nn.BatchNorm3d(648),
                                           nn.LeakyReLU(0.2))

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)

        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.shared_conv_1(x1)
        x2 = self.shared_conv_1(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.shared_conv_2(x1)
        x2 = self.shared_conv_2(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.shared_conv_3(x1)
        x2 = self.shared_conv_3(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.shared_conv_4(x1)
        x2 = self.shared_conv_4(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1 = self.class_1(x1)
        x2 = self.class_2(x2)

        x = torch.cat((x1, x2), dim=1)

        return self.classifier(x)


class shared_multimodality_three(nn.Module):
    def __init__(self):
        super(shared_multimodality_three, self).__init__()
        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_2 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_3 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.pool_1 = nn.MaxPool3d(2, 2)
        expansion = 2
        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_2_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_3_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_4_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_1_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_2_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_3_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_4_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_1_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_2_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_3_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_4_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1300 * 3, 500), nn.BatchNorm1d(500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.class_1 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(2048 * 5 * 6 * 5, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))
        self.class_2 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(2048 * 5 * 6 * 5, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))
        self.class_3 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(2048 * 5 * 6 * 5, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))

        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(2048 * 2, 1300),
        #                                 nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
        #                                 nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))
        # self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(648 * 5 * 6 * 5 * 2, 1300),
        #                                 nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
        #                                 nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.shared_conv_1 = nn.Sequential(residual_conv(32, 32), residual_conv(32, 32), nn.BatchNorm3d(32),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_2 = nn.Sequential(residual_conv(128, 128), residual_conv(128, 128), nn.BatchNorm3d(128),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_3 = nn.Sequential(residual_conv(512, 512), residual_conv(512, 512), nn.BatchNorm3d(512),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_4 = nn.Sequential(residual_conv(2048, 2048), residual_conv(2048, 2048), nn.BatchNorm3d(2048),
                                           nn.LeakyReLU(0.2))

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2, x3):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)
        x3_orig = self.conv_3(x3)

        x1 = self.res_block_1_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_1_3(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_1(x1)
        x2 = self.shared_conv_1(x2)
        x3 = self.shared_conv_1(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_2_3(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_2(x1)
        x2 = self.shared_conv_2(x2)
        x3 = self.shared_conv_2(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_3_3(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_3(x1)
        x2 = self.shared_conv_3(x2)
        x3 = self.shared_conv_3(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_4_3(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_4(x1)
        x2 = self.shared_conv_4(x2)
        x3 = self.shared_conv_4(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1 = self.class_1(x1)
        x2 = self.class_2(x2)
        x3 = self.class_3(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        return self.classifier(x)


#
# model = shared_multimodality_three()
# summary(model, ((4, 1, 160, 192, 160), (4, 1, 160, 192, 160), (4, 1, 160, 192, 160)))


class CosineLoss(nn.Module):
    def __init__(self, xent=0.1, reduction='mean'):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        self.y = torch.Tensor([1])

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y,
                                              reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)
        return cosine_loss + self.xent * cent_loss


class shared_multimodality_roi(nn.Module):
    def __init__(self, flatten_size):
        super(shared_multimodality_roi, self).__init__()
        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_2 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.pool_1 = nn.MaxPool3d(2, 2)
        expansion = 3

        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_2_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_3_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_4_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_1_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_2_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_3_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * 4
        self.res_block_4_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1300 * 2, 500), nn.BatchNorm1d(500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.class_1 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))
        self.class_2 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))

        self.shared_conv_1 = nn.Sequential(residual_conv(32, 32), residual_conv(32, 32), nn.BatchNorm3d(32),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_2 = nn.Sequential(residual_conv(128, 128), residual_conv(128, 128), nn.BatchNorm3d(128),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_3 = nn.Sequential(residual_conv(512, 512), residual_conv(512, 512), nn.BatchNorm3d(512),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_4 = nn.Sequential(residual_conv(2048, 2048), nn.BatchNorm3d(2048), nn.LeakyReLU(0.2))

        self.shared_att_1 = shared_att(2048, 2048)
        self.shared_att_2 = shared_att(2048, 2048)

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)

        x1 = self.res_block_1_1(x1_orig, x2_orig)
        x2 = self.res_block_1_2(x2_orig, x1_orig)
        x1 = self.shared_conv_1(x1)
        x2 = self.shared_conv_1(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_2_1(x1_orig, x2_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig)
        x1 = self.shared_conv_2(x1)
        x2 = self.shared_conv_2(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_3_1(x1_orig, x2_orig)
        x2 = self.res_block_3_2(x2_orig, x1_orig)
        x1 = self.shared_conv_3(x1)
        x2 = self.shared_conv_3(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.res_block_4_1(x1_orig, x2_orig)
        x2 = self.res_block_4_2(x2_orig, x1_orig)
        x1 = self.shared_conv_4(x1)
        x2 = self.shared_conv_4(x2)
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x1 = self.shared_att_1(x1_orig, x2_orig)
        x2 = self.shared_att_2(x2_orig, x1_orig)

        x1 = self.class_1(x1)
        x2 = self.class_2(x2)

        x = torch.cat((x1, x2), dim=1)

        return self.classifier(x)


class shared_att_three(nn.Module):
    def __init__(self, in_ch, img_ch):
        super(shared_att_three, self).__init__()
        self.att_1 = SASA_Layer(in_ch)
        self.att_2 = SASA_Layer(in_ch)
        self.res_1 = residual_conv(img_ch, img_ch)
        self.res_2 = residual_conv(img_ch, img_ch)
        self.norm = nn.BatchNorm3d(img_ch)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2, x3):
        x1 = self.res_1(x1)
        x1 = self.res_2(x1)
        x1 = self.norm(x1)
        x1 += self.att_1(x2)
        x1 += self.att_2(x3)
        return x1


class shared_att(nn.Module):
    def __init__(self, in_ch, img_ch):
        super(shared_att, self).__init__()
        self.att_1 = SASA_Layer(in_ch)
        self.att_2 = SASA_Layer(in_ch)
        self.res_1 = residual_conv(img_ch, img_ch)
        self.res_2 = residual_conv(img_ch, img_ch)
        self.norm = nn.BatchNorm3d(img_ch)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x1 = self.res_1(x1)
        x1 = self.res_2(x1)
        x1 = self.norm(x1)
        x1 += self.att_1(x2)
        return x1


class shared_multimodality_roi_three(nn.Module):
    def __init__(self, flatten_size):
        super(shared_multimodality_roi_three, self).__init__()
        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_2 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.conv_3 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])
        self.pool_1 = nn.MaxPool3d(2, 2)
        expansion = 3

        output_ch = 8
        self.res_block_1_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_1_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_1_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_1_4 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_2_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_2_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_2_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_2_4 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)

        output_ch = 8
        self.res_block_3_1 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_3_2 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_3_3 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)
        output_ch = output_ch * 5
        self.res_block_3_4 = Bottlenet_block_instance_three(output_ch, output_ch, expansion)

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1300 * 3, 500), nn.BatchNorm1d(500),
                                        nn.LeakyReLU(0.2), nn.Linear(500, 2))

        self.class_1 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))
        self.class_2 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))
        self.class_3 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                     nn.BatchNorm1d(1300), nn.LeakyReLU(0.2))

        self.shared_conv_1 = nn.Sequential(residual_conv(40, 40), residual_conv(40, 40), nn.BatchNorm3d(40),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_2 = nn.Sequential(residual_conv(200, 200), residual_conv(200, 200), nn.BatchNorm3d(200),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_3 = nn.Sequential(residual_conv(1000, 1000), residual_conv(1000, 1000), nn.BatchNorm3d(1000),
                                           nn.LeakyReLU(0.2))
        self.shared_conv_4 = nn.Sequential(residual_conv(5000, 5000), residual_conv(5000, 5000), nn.BatchNorm3d(5000),
                                           nn.LeakyReLU(0.2))
        self.shared_att_1 = shared_att_three(1000, 1000)
        self.shared_att_2 = shared_att_three(1000, 1000)
        self.shared_att_3 = shared_att_three(1000, 1000)

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x1, x2, x3):
        x1_orig = self.conv_1(x1)
        x2_orig = self.conv_2(x2)
        x3_orig = self.conv_3(x3)

        x1 = self.res_block_1_1(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_2_1(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_3_1(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_1(x1)
        x2 = self.shared_conv_1(x2)
        x3 = self.shared_conv_1(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_1_2(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_2_2(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_3_2(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_2(x1)
        x2 = self.shared_conv_2(x2)
        x3 = self.shared_conv_2(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_1_3(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_2_3(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_3_3(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_3(x1)
        x2 = self.shared_conv_3(x2)
        x3 = self.shared_conv_3(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_orig = x1.clone()
        x2_orig = x2.clone()
        x3_orig = x3.clone()
        x1 = self.res_block_1_4(x1_orig, x2_orig, x3_orig)
        x2 = self.res_block_1_4(x2_orig, x1_orig, x3_orig)
        x3 = self.res_block_1_4(x3_orig, x1_orig, x2_orig)
        x1 = self.shared_conv_4(x1)
        x2 = self.shared_conv_4(x2)
        x3 = self.shared_conv_4(x3)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        # x1 = self.shared_att_1(x1, x2, x3)
        # x2 = self.shared_att_1(x2, x1, x3)
        # x3 = self.shared_att_1(x3, x1, x2)

        x1 = self.class_1(x1)
        x2 = self.class_2(x2)
        x3 = self.class_3(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        return self.classifier(x)


class unimodal_roi(nn.Module):
    def __init__(self, flatten_size):
        super(unimodal_roi, self).__init__()
        expansion = 2
        output_ch = 8

        self.conv_1 = nn.Sequential(
            *[residual_conv(1, 2), residual_conv(2, 4), residual_conv(4, 8),
              nn.MaxPool3d(2, 2)])

        self.pool = nn.MaxPool3d(2, 2)
        expansion = 2

        self.res_block_1 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * expansion + output_ch
        self.res_block_2 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * expansion + output_ch
        self.res_block_3 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        output_ch = output_ch * expansion + output_ch
        # self.res_block_4 = Bottlenet_block_instance(output_ch, output_ch, expansion)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(flatten_size, 1300),
                                        nn.BatchNorm1d(1300), nn.LeakyReLU(0.2), nn.Linear(1300, 500),
                                        nn.BatchNorm1d(500), nn.LeakyReLU(0.2), nn.Linear(500, 2))

    def forward(self, x):
        x = self.conv_1(x)

        x = self.res_block_1(x, x)
        x = self.pool(x)

        x = self.res_block_2(x, x)
        x = self.pool(x)

        x = self.res_block_3(x, x)
        x = self.pool(x)
        # x = self.res_block_4(x, x)
        # x = self.pool(x)

        return self.classifier(x)

# model = shared_multimodality_roi(2048 * 2 * 3 * 2)
# summary(model, ((4, 1, 80, 96, 80), (4, 1, 80, 96, 80)))
# model = shared_multimodality_roi(2048 * 1 * 1 * 1)
# summary(model, ((4, 1, 50, 50, 50), (4, 1, 50, 50, 50)))

# model = shared_multimodality_roi_three(5000 * 2 * 3 * 2)
# summary(model, ((4, 1, 80, 96, 80), (4, 1, 80, 96, 80), (4, 1, 80, 96, 80)))
# model = shared_multimodality_roi_three(5000 * 1 * 1 * 1)
# summary(model, ((4, 1, 50, 50, 50), (4, 1, 50, 50, 50), (4, 1, 50, 50, 50)))

# model = unimodal_roi(216 * 5 * 6 * 5)
# summary(model, (4, 1, 50, 50, 50))
# summary(model, (4, 1, 80, 96, 80))
# model = residual_first_wide_multi()
# summary(model, ((4, 1, 160, 196, 160), (4, 1, 160, 196, 160)))
# model = residual_first_wide_mixing()
# summary(model, ((4, 1, 160, 196, 160), (4, 1, 160, 196, 160)))
# model = residual_first_wide_unimodal()
# summary(model, (4, 1, 160, 196, 160))
# model = shared_multimodality()
# summary(model, ((4, 1, 160, 196, 160), (4, 1, 160, 196, 160)))
# model = shared_multimodality_wide()
# summary(model, ((4, 1, 160, 196, 160), (4, 1, 160, 196, 160)))

## we have to change from batch norm to instance norm
