import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=198):
        super().__init__()
        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, query, src, src_mask, flag):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]
        # batch_size = src.shape[0]
        # src_len = src.shape[1]
        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # pos = [batch size, src len]
        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]
        if flag == 0:
            for layer in self.layers:
                src1, att = layer(query, src, src_mask)

        if flag == 1:
            for layer in self.layers:
                src1, att = layer.guide(query, src, src_mask)

        # src = [batch size, src len, hid dim]
        return src1, att


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attn_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm2 = nn.LayerNorm(hid_dim)
        # self.ff_layer_norm3 = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, query, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        _src, att = self.self_attention(query, src, src, src_mask)
        # print(_src.size(), src.size())
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src, att

    def guide(self, query, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        _src, att = self.self_attention(query, src, src, src_mask)
        _src = self.self_attn_layer_norm(self.dropout(_src))
        _src = self.positionwise_feedforward(_src)
        _src = self.ff_layer_norm(self.dropout(_src))
        return _src, att


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
        self.l1 = nn.Linear(hid_dim * 2, hid_dim)
        self.l2 = nn.Linear(hid_dim * 2, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.p = nn.Parameter(torch.FloatTensor(1, self.hid_dim))

    def forward(self, query, key, value, mask=None):
        # print('query:', query.size())
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]
        if mask is not None:
            # _MASKING_VALUE=-1e10

            _MASKING_VALUE = -1e10 if energy.dtype == torch.float32 else -1e+4
            energy = energy.masked_fill(mask == 0, _MASKING_VALUE)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]
        # x = torch.matmul(self.dropout(attention), V)
        x = torch.matmul(attention, V)
        # x= [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)

        # print('att_result:', x.size())
        # print('query:', query.size())
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        x = torch.cat([x, query], dim=2)
        # print('x:', x.size())
        x1 = self.sigmoid(self.l1(x))
        x2 = self.l2(x)
        x = torch.mul(x1, x2)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]
        x = torch.relu(self.fc_2(x))
        # x = [batch size, seq len, hid dim]
        return x


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # self.decoder_att1 = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, encoder_out, mask, decoder_hidden, mask_need):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)

        att = self.full_att(self.relu(att1 * att2.unsqueeze(1))).squeeze(2)
        if mask_need == 1:
            att = att * mask
            alpha = self.softmax(att) * mask

        if mask_need == 0:
            att = att
            alpha = self.softmax(att)
        # alpha = alpha * mask
        # print(encoder_out.size())
        # print(alpha.unsqueeze(2).size(), att.size())
        # print(mask)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
        self.gating = gating

    def forward(self, x):
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        x = F.normalize(x)

        return x


# ============================================================
class Fuxiao(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.line1 = nn.Linear(1024, 1)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.line4 = nn.Linear(1536, 512)
        self.line5 = nn.Linear(1536, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.encoder1 = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder2 = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder3 = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder4 = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder5 = Encoder(512, 512, 1, 8, 512, 0.1)
        # self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=False, dropout=0.1)
        self.lstm1 = nn.LSTM(512, 512, batch_first=True, bidirectional=False, dropout=0.1)
        self.pos_embedding = nn.Embedding(15, 512)
        self.type_embedding = nn.Embedding(3, 512)
        self.loss = nn.BCEWithLogitsLoss()
        self.cls_embedding = nn.Embedding(1, 512)
        self.count_embedding = nn.Embedding(16, 512)
        self.count_embedding1 = nn.Embedding(16, 512)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, image_features, caption_features, caption_g_features, text_features1, text_features2, text_features3, count_list1, max_score1, secid, imageid,
                mask_section, logit_scale):
        # image patches
        # print(count_list1.size())
        #print(text_features1.size(), text_features2.size(), text_features3.size())

        #print(caption_features.size(), caption_g_features.size())

        mask_image1 = torch.ones(image_features.size()[0], 50).float().to(device).unsqueeze(1).unsqueeze(1)
        image_features, _ = self.encoder2(image_features, image_features, mask_image1, 0)
        mask_caption1 = torch.ones(image_features.size()[0], 77).float().to(device).unsqueeze(1).unsqueeze(1)
        caption_features, _ = self.encoder3(caption_features, caption_features, mask_caption1, 0)
        #caption_g_features, _ = self.encoder3(caption_g_features, caption_g_features, mask_caption1, 0)

        mask_image = torch.ones(image_features.size()[0], 50).float().to(device)
        mask_caption = torch.ones(image_features.size()[0], 77).float().to(device)
        fusion_feature = torch.zeros(image_features.size()[0], 1).long().to(device)
        fusion_feature = self.cls_embedding(fusion_feature)
        caption_features_tmp = torch.cat([fusion_feature, image_features, caption_features], dim=1)
        mask_i_c_tmp = torch.ones(mask_section.size()[0], 1).float().to(device)
        mask_fus = torch.cat([mask_i_c_tmp, mask_image, mask_caption], dim=1).unsqueeze(1).unsqueeze(1)
        caption_features1, _ = self.encoder4(caption_features_tmp, caption_features_tmp, mask_fus, 0)
        caption_features= caption_features1[:, 0, :]





        # caption_features = torch.cat([caption_features, image_features], dim=-1)
        # caption_features = self.dropout(self.relu(self.line2(caption_features)))

        batch_size = image_features.size(0)
        text_features1_new11 = []
        text_features1_new22 = []
        text_features1_new33 = []

        # all the text and images are using patches and tokens level
        for i in range(batch_size):
            mask_caption = torch.ones(15, 77).float().to(device).unsqueeze(1).unsqueeze(1)
            text_features1_tmp1, _ = self.encoder3(text_features1[i, :, :], text_features1[i, :, :], mask_caption, 0)
            text_features1_tmp1 = text_features1_tmp1[:, -1, :]

            text_features1_tmp2, _ = self.encoder3(text_features2[i, :, :], text_features2[i, :, :], mask_caption, 0)
            text_features1_tmp2 = text_features1_tmp2[:, -1, :]

            text_features1_tmp3, _ = self.encoder3(text_features3[i, :, :], text_features3[i, :, :], mask_caption, 0)
            text_features1_tmp3 = text_features1_tmp3[:, -1, :]

            #text_features1_tmp = torch.cat([text_features1_tmp1, text_features1_tmp2, text_features1_tmp3], dim=-1)
           # text_features1_tmp = self.dropout(self.relu(self.line4(text_features1_tmp)))

            text_features1_new11.append(text_features1_tmp1.unsqueeze(0))
            text_features1_new22.append(text_features1_tmp2.unsqueeze(0))
            text_features1_new33.append(text_features1_tmp3.unsqueeze(0))

        text_features1 = torch.cat(text_features1_new11, dim=0)
        text_features2 = torch.cat(text_features1_new22, dim=0)
        text_features3 = torch.cat(text_features1_new33, dim=0)

        #text_features = text_features1

        # This is for the mask
        mask_i_c = torch.ones(mask_section.size()[0], 1).float().to(device)
        mask = torch.cat([mask_i_c, mask_section, mask_section, mask_section], dim=1).unsqueeze(1).unsqueeze(1)

        # self.count_embedding
        count_list1_embedding = self.count_embedding(count_list1)
        count_list1_embedding1 = self.count_embedding(max_score1)

        ###Type Embedding
        image_type = torch.zeros(image_features.size()[0]).long().to(device)
        image_type_embed = self.type_embedding(image_type)
        caption_type = torch.ones(caption_features.size()[0]).long().to(device)
        caption_type_embed = self.type_embedding(caption_type)
        section_type = 2 * torch.ones(text_features1.size()[0], text_features1.size()[1]).long().to(device)
        section_type_embed = self.type_embedding(section_type)

        ##Position Embedding
        batch_size = text_features1.size(0)
        pos_text_embed = self.pos_embedding(
            torch.arange(0, text_features1.size(1)).unsqueeze(0).repeat(batch_size, 1).to(device))
        pos_text_embed, _ = self.lstm(pos_text_embed)

        secid = secid.long()
        imageid = imageid.long()
        # pos_image_embed = self.pos_embedding(torch.arange(0, img_set_features1.size(1)).unsqueeze(0).repeat(batch_size, 1).to(device))
        pos_image_embed = self.pos_embedding(imageid).to(device)
        # pos_image_embed,_ = self.lstm1(pos_image_embed)
        # new embed
        #image_features = image_features + image_type_embed + pos_image_embed
        # image_features = img_set_features1 + image_type_embed  + pos_image_embed
        #caption_features = caption_features + caption_type_embed + pos_image_embed
        #text_features = text_features + section_type_embed + pos_text_embed + count_list1_embedding + count_list1_embedding1
        caption_features = caption_features + caption_type_embed + pos_image_embed
        text_features1 = text_features1 + section_type_embed + pos_text_embed + count_list1_embedding + count_list1_embedding1
        text_features2 = text_features2 + section_type_embed + pos_text_embed + count_list1_embedding + count_list1_embedding1
        text_features3 = text_features3 + section_type_embed + pos_text_embed + count_list1_embedding + count_list1_embedding1
        # text_features = text_features + section_type_embed + pos_text_embed

        features = torch.cat([caption_features.unsqueeze(1), text_features1, text_features2, text_features3], dim=1)
        # features = torch.cat([caption_features.unsqueeze(1), text_features], dim=1)
        features_new, _ = self.encoder1(features, features, mask, 0)

        #image_features1 = features_new[:, :1, :]
        caption_features1 = features_new[:, :1, :]
        text_features1 = features_new[:, 1:16, :]
        text_features2 = features_new[:, 16:31, :]
        text_features3 = features_new[:, 31:, :]
        # image_features1 = features_new[:, :1, :]
        # caption_features1 = features_new[:, :1, :]
        # text_features1 = features_new[:, 1:, :]
        #text_features1 = self.dropout(self.relu(self.line5(torch.cat([text_features1, text_features2, text_features3], dim=-1))))

        # calculate the similarity score
        batch_size = text_features1.size(0)
        score_list = []
        total_loss = 0
        metrics1, metrics3 = 0, 0
        num_of_image = 0
        article_acc = []
        labels = secid

        logits_per_image1_new = []
        for i in range(batch_size):
            #tmp1 = logit_scale * image_features1[i, :, :] @ text_features1[i, :, :].T
            tmp2 = logit_scale * caption_features1[i, :, :] @ text_features1[i, :, :].T

            #tmp11 = logit_scale * image_features1[i, :, :] @ text_features2[i, :, :].T
            tmp21 = logit_scale * caption_features1[i, :, :] @ text_features2[i, :, :].T

            #tmp12 = logit_scale * image_features1[i, :, :] @ text_features3[i, :, :].T
            tmp22 = logit_scale * caption_features1[i, :, :] @ text_features3[i, :, :].T
            # tmp1 = self.cos(image_features1[i, :, :],text_features1[i, :, :])
            # tmp2 = self.cos(caption_features1[i, :, :], text_features1[i, :, :])
            '''
            caption_features_hard1 = torch.cat([caption_features1[:i, :, :], caption_features1[i+1:, :, :]], dim=0).squeeze(1)
            text_feature_hard1 = text_features1[i, labels[i], :].unsqueeze(0)
            hard_neg_cap1 = logit_scale * text_feature_hard1 @ caption_features_hard1.T
            hard_neg_img1 = (logit_scale * text_feature_hard1 @ image_features1[i, :, :].T).repeat(1, 31)

            img_features_hard1 = torch.cat([image_features1[:i, :, :], image_features1[i+1:, :, :]], dim=0).squeeze(1)
            hard_neg_cap2 = (logit_scale * text_feature_hard1 @ caption_features1[i, :, :].T).repeat(1, 31)
            hard_neg_img2 = logit_scale * text_feature_hard1 @ img_features_hard1.T
            '''
            mask_new = mask_section[i].unsqueeze(0)

            #logits_per_image11 = tmp1 * mask_new
            logits_per_image22 = tmp2 * mask_new

            #logits_per_image12 = tmp11 * mask_new
            logits_per_image23 = tmp21 * mask_new

            #logits_per_image13 = tmp12 * mask_new
            logits_per_image24 = tmp22 * mask_new
            # logits_per_image1 = logits_per_image11
            # a = self.dropout(self.relu(self.line1(torch.cat([image_features1[i, :, :], caption_features1[i, :, :]], dim=1))))
            # logits_per_image_v = a * logits_per_image11 + (1-a) * logits_per_image22
            # logits_per_image_v = logits_per_image22
            #logits_per_image_v = torch.maximum(logits_per_image11, logits_per_image22)
            #logits_per_image_v_tmp1 = torch.maximum(logits_per_image11, logits_per_image12)
            #logits_per_image_v_tmp2 = torch.maximum(logits_per_image_v_tmp1, logits_per_image13)

            logits_per_image_t_tmp1 = torch.maximum(logits_per_image22, logits_per_image23)
            logits_per_image_t_tmp2 = torch.maximum(logits_per_image_t_tmp1, logits_per_image24)
            #logits_per_image_t_tmp2 = logits_per_image22
            #logits_per_image_v = torch.maximum(logits_per_image_v_tmp2, logits_per_image_t_tmp2)
            logits_per_image_v = logits_per_image_t_tmp2
            # logits_per_image_v = 0.5 * logits_per_image11 + 0.5 * logits_per_image22

            # print(logits_per_image_v.size())
            # logits_per_image_v = logits_per_image11

            logits_per_image1_new.append(logits_per_image_v)

        logits_per_image1 = torch.cat(logits_per_image1_new, dim=0)
        # logits_per_image1 = torch.maximum(logits_per_image1, max_score1)
        # print('logits_per_image1:', logits_per_image1.size())
        # print('labels:', labels.size())
        # print('max_score:', max_score1.size())
        loss = (F.cross_entropy(logits_per_image1, labels))
        total_loss = total_loss + loss

        # calculate prediction result
        ranking = torch.argsort(logits_per_image1, descending=True, dim=1)
        ground_truth = labels.reshape(labels.size(0), 1)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics1 += np.sum(preds < 1)
        metrics3 += np.sum(preds < 3)
        num_of_image1 = len(preds)
        num_of_image += num_of_image1
        return total_loss, metrics1, metrics3, num_of_image, ranking.detach().cpu().numpy(), ground_truth.detach().cpu().numpy()

