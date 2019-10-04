from ..document_bert import BertForDocumentClassification
from .util import get_model_path
class ObesityPhenotypingBert(BertForDocumentClassification):
    def __init__(self, device='cuda', batch_size=10, model_name="n2c2_2008_obesity_lstm"):
        model_path = get_model_path(model_name)
        self.labels = "Gout, Venous Insufficiency, Gallstones, Hypertension, Obesity, Asthma, GERD, Hypercholesterolemia, Hypertriglyceridemia, CHF, OSA, OA, PVD, CAD, Depression, Diabetes".split(', ')

        super().__init__(device=device,
                         batch_size=batch_size,
                         bert_batch_size=7,
                         bert_model_path=model_path,
                         architecture='DocumentBertLSTM',
                         labels=self.labels)