from ..document_bert import BertForDocumentClassification
from .util import get_model_path
class SmokerPhenotypingBert(BertForDocumentClassification):
    def __init__(self, device='cuda', batch_size=10, model_name="n2c2_2006_smoker_lstm"):
        model_path = get_model_path(model_name)
        self.labels = "PAST SMOKER, CURRENT SMOKER, NON-SMOKER, UNKNOWN".split(', ')

        super().__init__(device=device,
                         batch_size=batch_size,
                         bert_batch_size=7,
                         bert_model_path=model_path,
                         architecture='DocumentBertLSTM',
                         labels=self.labels)