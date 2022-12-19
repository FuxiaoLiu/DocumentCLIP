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
# the main function
class Fuxiao(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.encoder_image = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder_caption = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder_earlyfusion = Encoder(512, 512, 1, 8, 512, 0.1)
        self.encoder_sentence = Encoder(512, 512, 1, 8, 512, 0.1)
        self.layout_transformer = Encoder(512, 512, 1, 8, 512, 0.1)
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=False, dropout=0.1)
        self.pos_embedding = nn.Embedding(15, 512)
        self.type_embedding = nn.Embedding(3, 512)
        self.loss = nn.BCEWithLogitsLoss()
        self.cls_embedding = nn.Embedding(1, 512)
        self.count_embedding = nn.Embedding(16, 512)


    def forward(self, image_features, caption_features, section_sen1, section_sen2, section_sen3, count_list_entity, max_score_clip, secid, imageid, mask_section, logit_scale):
        
        # Encode Image from patches
        mask_image = torch.ones(image_features.size()[0], 50).float().to(device).unsqueeze(1).unsqueeze(1)
        image_features, _ = self.encoder_image(image_features, image_features, mask_image, 0)
        
        
        # Encode Caption from tokens
        mask_caption = torch.ones(image_features.size()[0], 77).float().to(device).unsqueeze(1).unsqueeze(1)
        caption_features, _ = self.encoder_caption(caption_features, caption_features, mask_caption, 0)

        
        # early fusion of image and caption
        mask_image = torch.ones(image_features.size()[0], 50).float().to(device)
        mask_caption = torch.ones(image_features.size()[0], 77).float().to(device)
        fusion_feature = torch.zeros(image_features.size()[0], 1).long().to(device)
        fusion_feature = self.cls_embedding(fusion_feature)
        caption_features_tmp = torch.cat([fusion_feature, image_features, caption_features], dim=1)
        mask_i_c_tmp = torch.ones(mask_section.size()[0], 1).float().to(device)
        mask_fus = torch.cat([mask_i_c_tmp, mask_image, mask_caption], dim=1).unsqueeze(1).unsqueeze(1)
        IC_features, _ = self.encoder_earlyfusion(caption_features_tmp, caption_features_tmp, mask_fus, 0)
        IC_features= IC_features[:, 0, :]


        batch_size = image_features.size(0)
        sec_sen1_features_list = []
        sec_sen2_features_list = []
        sec_sen3_features_list = []

        # section sentence encoding
        batch_size = caption_features.size(0)
        for i in range(batch_size):
            mask_sen = torch.ones(15, 77).float().to(device).unsqueeze(1).unsqueeze(1)
            sec_features1, _ = self.encoder_sentence(section_sen1[i, :, :], section_sen1[i, :, :], mask_sen, 0)

            sec_features2, _ = self.encoder_sentence(section_sen2[i, :, :], section_sen2[i, :, :], mask_sen, 0)

            sec_features3, _ = self.encoder_sentence(section_sen3[i, :, :], section_sen3[i, :, :], mask_sen, 0)

            sec_sen1_features_list.append(sec_features1[:, -1, :].unsqueeze(0))
            sec_sen2_features_list.append(sec_features2[:, -1, :].unsqueeze(0))
            sec_sen3_features_list.append(sec_features3[:, -1, :].unsqueeze(0))

        sec_sen1_features = torch.cat(sec_sen1_features_list, dim=0)
        sec_sen2_features = torch.cat(sec_sen2_features_list, dim=0)
        sec_sen3_features = torch.cat(sec_sen3_features_list, dim=0)


        
        # The following is discussing position, segment and entity embeddings for the layout transformer
        # entity sorting embedding
        entity_sorting_embedding = self.count_embedding(count_list_entity)
        
        # clip feature sorting embedding
        clip_sorting_embedding = self.count_embedding(max_score_clip)

        # Segment Embedding
        image_type = torch.zeros(image_features.size()[0]).long().to(device)
        image_type_embed = self.type_embedding(image_type)
        caption_type = torch.ones(caption_features.size()[0]).long().to(device)
        caption_type_embed = self.type_embedding(caption_type)
        section_type = 2 * torch.ones(text_features1.size()[0], text_features1.size()[1]).long().to(device)
        section_type_embed = self.type_embedding(section_type)

        # Section position Embedding
        pos_section_embed = self.pos_embedding(torch.arange(0, sec_sen1_features.size(1)).unsqueeze(0).repeat(batch_size, 1).to(device))
        pos_section_embed, _ = self.lstm(pos_section_embed)

        # Image/caption position Embedding
        secid = secid.long()
        imageid = imageid.long()
        pos_image_embed = self.pos_embedding(imageid).to(device)
        pos_caption_embed = pos_image_embed
        
    
        # Aggregation of the position embedding, segment embedding, entity embedding before the layout transformer

        IC_features = IC_features + caption_type_embed + pos_caption_embed
        sec_sen1_features = sec_sen1_features + section_type_embed + pos_section_embed + entity_sorting_embedding + clip_sorting_embedding
        sec_sen2_features = sec_sen2_features + section_type_embed + pos_section_embed + entity_sorting_embedding + clip_sorting_embedding
        sec_sen3_features = sec_sen3_features + section_type_embed + pos_section_embed + entity_sorting_embedding + clip_sorting_embedding
        
        # sec_sen1_features, sec_sen2_features, sec_sen3_features are the top3 candidates from each section

        
        # mask
        mask_i_c = torch.ones(mask_section.size()[0], 1).float().to(device)
        mask = torch.cat([mask_i_c, mask_section, mask_section, mask_section], dim=1).unsqueeze(1).unsqueeze(1)
        
        features_fuse = torch.cat([IC_features.unsqueeze(1), sec_sen1_features, sec_sen2_features, sec_sen3_features], dim=1)
        features_contextual, _ = self.layout_transformer(features_fuse, features_fuse, mask, 0)
        
        # Contextual Embeddings after layout transformer
        contextual_IC_feature = features_contextual[:, :1, :]
        contextual_sen1ofsection_feature = features_contextual[:, 1:16, :]
        contextual_sen2ofsection_feature = features_contextual[:, 16:31, :]
        contextual_sen3ofsection_feature = features_contextual[:, 31:, :]


        # calculate the cosine similarity score between image/caption pairs with each sentence in the section
        Cos_score_list= []
        for i in range(batch_size):
            cos_score1 = logit_scale * contextual_IC_feature[i, :, :] @ contextual_sen1ofsection_feature[i, :, :].T
            cos_score2 = logit_scale * contextual_IC_feature[i, :, :] @ contextual_sen2ofsection_feature[i, :, :].T
            cos_score3 = logit_scale * contextual_IC_feature[i, :, :] @ contextual_sen3ofsection_feature[i, :, :].T
            mask_new = mask_section[i].unsqueeze(0)
            cos_score1 = cos_score1 * mask_new
            cos_score2 = cos_score2 * mask_new
            cos_score3 = cos_score3 * mask_new

            # "Max" Strategy in our paper
            Max_cos_score = torch.maximum(cos_score1, cos_score2)
            Max_cos_score = torch.maximum(Max_cos_score, cos_score3)
            Cos_score_list.append(Max_cos_score)

        prediction_score = torch.cat(Cos_score_list, dim=0)
        
        # Contrastive Learning Loss
        labels = secid
        total_loss = 0
        loss = (F.cross_entropy(prediction_score, labels))
        total_loss = total_loss + loss

        # calculate prediction evaluation metirc: R@1 and R@3
        R1, R3 = 0, 0
        #ranking the prediction score
        ranking = torch.argsort(prediction_score, descending=True, dim=1)
        ground_truth = labels.reshape(labels.size(0), 1)
        #compare with groundtruth
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        # calculate R@1 and R@3
        R1 += np.sum(preds < 1)
        R3 += np.sum(preds < 3)
        num_of_image += len(preds)
        
        return total_loss, R1, R3, num_of_image, ranking.detach().cpu().numpy(), ground_truth.detach().cpu().numpy()

