from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from torch import nn
import torch
from .transformer import TransformerEncoderLayer, TransformerEncoder

from torch.nn import LSTM
class DocumentBertLSTM(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, freeze_bert=False):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(False):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        output, (_, _) = self.lstm(bert_output.permute(1,0,2))

        last_layer = output[-1]
        #print("Last LSTM layer shape:",last_layer.shape)

        prediction = self.classifier(last_layer)
        #print("Prediction Shape", prediction.shape)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertLinear(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * self.bert_batch_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, freeze_bert=False):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(False):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertMaxPool(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertMaxPool, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        # self.transformer_encoder = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
        #                                            nhead=6,
        #                                            dropout=bert_model_config.hidden_dropout_prob)
        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, freeze_bert=False):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(False):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertTransformer(BertPreTrainedModel):
    """
    BERT -> TransformerEncoder -> Max over attention output.
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertTransformer, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        encoder_layer = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
                                                   nhead=6,
                                                   dropout=bert_model_config.hidden_dropout_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, freeze_bert=True):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(False):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        transformer_output = self.transformer_encoder(bert_output.permute(1,0,2))

        #print(transformer_output.shape)

        prediction = self.classifier(transformer_output.permute(1,0,2).max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction