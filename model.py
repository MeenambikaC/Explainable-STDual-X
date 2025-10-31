import torch


class PositionalEncoding(nn.Module):
    def __init__(self, k, d_model, seq_len):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros([k, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(seq_len)], requires_grad=False).unsqueeze(1).repeat(1, k)
        s = 0.0
        
        interval = seq_len / k
        mu = []
        for _ in range(k):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(k)]).unsqueeze(0))
        
    def normal_pdf(self, pos, mu, sigma):
        a = pos - mu
        log_p = -1*torch.mul(a, a)/(2*(sigma**2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(pdfs, self.embedding)
        
        return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, _heads, dropout, seq_len):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self._attention = nn.MultiheadAttention(seq_len, _heads, batch_first=True)
        
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.cnn_units = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_units, (1, 1)),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, self.cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src = self.attn_norm(src + self.attention(src, src, src)[0] + self._attention(src.transpose(-1, -2), src.transpose(-1, -2), src.transpose(-1, -2))[0].transpose(-1, -2))
        
        src = self.final_norm(src + self.cnn(src.unsqueeze(dim=1)).squeeze(dim=1))
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, _heads, seq_len, num_layer=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(TransformerEncoderLayer(d_model, heads, _heads, dropout, seq_len))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src

class Transformer(nn.Module):
    def __init__(self, num_layer, d_model, k, heads, _heads, seq_len, trg_len, dropout):
        super(Transformer, self).__init__()
        

        self.pos_encoding = PositionalEncoding(k, d_model, seq_len)

        self.encoder = TransformerEncoder(d_model, heads, _heads, seq_len, num_layer, dropout)

    def forward(self, inputs):
        encoded_inputs = self.pos_encoding(inputs)

        return self.encoder(encoded_inputs)

class Model(nn.Module):
    def __init__(self, feature_count, l, trg_len, num_classes):
        super(Model, self).__init__()
        
        
        self.imu_transformer = Transformer(3, feature_count, 100, 4, 4, l, trg_len, 0.1)
        
        self.linear_imu = nn.Sequential(
            nn.Linear(feature_count*l, (feature_count*l)//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear((feature_count*l)//2, trg_len),
            nn.ReLU()
        )
        
        # Batch normalization and dropout layers
        self.batch_norm = nn.BatchNorm1d(trg_len)
        self.dropout = nn.Dropout(0.5)
        


    def forward(self, inputs):
        
        embedding = self.linear_imu(torch.flatten(self.imu_transformer(inputs), start_dim=1, end_dim=2))
        
        # Apply batch normalization
        embedding = self.batch_norm(embedding)
        
        # Apply dropout
        embedding = self.dropout(embedding)
        

        
        return embedding

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x,y):
        queries = self.query(x)
        keys = self.key(y)
        values = self.value(y)
        queries_1=queries.unsqueeze(1)
        keys_1=keys.unsqueeze(1)
        values_1=values.unsqueeze(1)
        scores = torch.bmm(queries_1, keys_1.transpose(1, 2)) / (self.input_dim ** 0.5)
        
        return scores, values_1


class CrossSensorAttention(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(CrossSensorAttention,self).__init__()
        self.input_dim=input_dim
        self.softmax =nn.Softmax(dim=1)
        self.attention_cal =Attention(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self,emb_1,emb_2,emb_3):
        attention_weight_1_2,val_1_2 = self.attention_cal(emb_1,emb_2)
        attention_weight_1_3,val_1_3 = self.attention_cal(emb_1,emb_3)
        attention_weight_2_1,val_2_1 = self.attention_cal(emb_2,emb_1)
        attention_weight_2_3,val_2_3 = self.attention_cal(emb_2,emb_3)
        attention_weight_3_1,val_3_1 = self.attention_cal(emb_3,emb_1)
        attention_weight_3_2,val_3_2 = self.attention_cal(emb_3,emb_2)
        # print(attention_weight_1_2.shape,"attention_weight_1_2")
        stacked_attention = torch.stack([attention_weight_1_2.squeeze(1),attention_weight_1_3.squeeze(1),attention_weight_2_1.squeeze(1),attention_weight_2_3.squeeze(1),attention_weight_3_1.squeeze(1),attention_weight_3_2.squeeze(1)],dim=1)
        # print(stacked_attention.shape,"stacked_attention")
        attention = self.softmax(stacked_attention)
        # print(attention[32],"att_val")
        # print(attention[0].sum(),"att_sum")
        # print(attention.shape,"attention")
        attention_matrix=attention.unsqueeze(2)
        attention_matrix=attention_matrix.transpose(0,1)
        # print(attention_matrix.shape,"attention_matrix")
        attention_matrix_1=attention_matrix[0]
        attention_matrix_2=attention_matrix[1]
        attention_matrix_3=attention_matrix[2]
        attention_matrix_4=attention_matrix[3]
        attention_matrix_5=attention_matrix[4]
        attention_matrix_6=attention_matrix[5]
        attention_matrices = [attention_matrix_1, attention_matrix_2, attention_matrix_3, attention_matrix_4, attention_matrix_5, attention_matrix_6]
        values = [val_1_2, val_1_3, val_2_1, val_2_3, val_3_1, val_3_2]
        # print(attention_matrix_1.shape,"attention_matrix_1")
        # print(val_1_2.shape,"val_1_2")
        output_values = []
        for attn, val in zip(attention_matrices, values):
            out=torch.bmm(attn, val)
            output_val=out.squeeze(dim=1)
            output_values.append(output_val)
        
        final_embedding = emb_1+emb_2+emb_3
        for val in output_values:
            final_embedding+=val
        class_scores =self.classifier(final_embedding)
        
        return class_scores,final_embedding,attention_matrix


class MultiSensorModel(nn.Module):
    def __init__(self, feature_count, l, trg_len, num_classes, feature_dim):
        super(MultiSensorModel, self).__init__()
        
        # Three separate models for each sensor placement
        self.sensor_model_1 = Model(feature_count, l, trg_len, num_classes)
        # Cross-attention layer between embeddings
        self.cross_sensor_attention = CrossSensorAttention(feature_dim, num_classes)

    def forward(self, sensor_data1, sensor_data2,sensor_data3):
        # Get embeddings from each sensor model
        embedding_1 = self.sensor_model_1(sensor_data1)
        embedding_2 = self.sensor_model_1(sensor_data2)
        embedding_3 = self.sensor_model_1(sensor_data3)

        class_scores, feature_embedding,attention_matrix = self.cross_sensor_attention(embedding_1, embedding_2,embedding_3)
        
        # Return class scores and feature embeddings
        return class_scores, feature_embedding, embedding_1, embedding_2,embedding_3,attention_matrix
