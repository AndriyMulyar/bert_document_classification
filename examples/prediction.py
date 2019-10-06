from bert_document_classification.models import ObesityPhenotypingBert

obesity_classifier = ObesityPhenotypingBert(device='cuda', batch_size=10) #defaults to GPU prediction

documents = [
    "I'm a document! Even if i'm long this model will perform well!",
    "Another one! The patient has hypertension."

]

predictions = obesity_classifier.predict(documents).transpose(0,1)

for prediction in predictions:
    print({obesity_classifier.labels[i]: prediction[i].item() for i in range(len(obesity_classifier.labels))})


"""
Expected output:
{'Gout': 0.0, 'Venous Insufficiency': 1.0, 'Gallstones': 0.0, 'Hypertension': 1.0, 'Obesity': 0.0, 'Asthma': 1.0, 'GERD': 1.0, 'Hypercholesterolemia': 0.0, 'Hypertriglyceridemia': 0.0, 'CHF': 0.0, 'OSA': 0.0, 'OA': 0.0, 'PVD': 0.0, 'CAD': 0.0, 'Depression': 0.0, 'Diabetes': 0.0}
{'Gout': 0.0, 'Venous Insufficiency': 1.0, 'Gallstones': 1.0, 'Hypertension': 1.0, 'Obesity': 1.0, 'Asthma': 0.0, 'GERD': 0.0, 'Hypercholesterolemia': 1.0, 'Hypertriglyceridemia': 0.0, 'CHF': 0.0, 'OSA': 0.0, 'OA': 1.0, 'PVD': 0.0, 'CAD': 0.0, 'Depression': 1.0, 'Diabetes': 0.0}
Of-course, one requires documents that are actual EHR's for these predictions to be meaningful.
"""