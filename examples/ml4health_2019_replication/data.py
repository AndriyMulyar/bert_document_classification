from pkg_resources import resource_exists, resource_listdir, resource_string, resource_stream,resource_filename
import xml.etree.ElementTree as ET
import numpy



def load_n2c2_2006_train_dev_split():
    train = list(load_n2c2_2006(partition='train'))
    numpy.random.seed(0)
    numpy.random.shuffle(train)

    labels = {}
    for id, doc, label in train:
        if label not in labels:
            labels[label] = []
        labels[label].append(tuple((id,doc,label)))

    dev = []
    train = []
    for label in labels.keys():
        dev += labels[label][:int(len(labels[label])*.2)]
        train += labels[label][int(len(labels[label])*.2):]

    return train,dev

def load_n2c2_2006(partition='train'):
    """
    Yields a generator of id, doc, label tuples.
    :param partition:
    :return:
    """
    assert partition in ['train', 'test']

    with open("data/smokers_surrogate_%s_all_version2.xml" % partition) as raw:
        file = raw.read().strip()
    # file = resource_string('clinical_data', 'phenotyping/n2c2_2006/smokers_surrogate_%s_all_version2.xml' % partition).decode('utf-8').strip()
    root = ET.fromstring(file)
    ids = []
    notes = []
    labels = []
    documents = root.findall("./RECORD")
    for document in documents:
        ids.append(document.attrib['ID'])
        notes.append(document.findall('./TEXT')[0].text)
        labels.append(document.findall('./SMOKING')[0].attrib['STATUS'])

    for id, note, label in zip(ids,notes,labels):
        yield (id,note,label)


def load_n2c2_2008_train_dev_split():
    train = list(load_n2c2_2008(partition='train'))

    return train[:int(len(train)*.8)], train[int(len(train)*.8):]


def load_n2c2_2008(partition='train'):
    assert partition in ['train', 'test']
    documents = {} #id : text
    all_diseases = set()
    notes = tuple()
    if partition == 'train':
        with open('data/obesity_patient_records_training.xml') as t1, \
                open('data/obesity_patient_records_training2.xml') as t2:
            notes1 = t1.read().strip()
            notes2 = t2.read().strip()
        notes = (notes1,notes2)
    elif partition == 'test':
        with open('data/obesity_patient_records_test.xml') as t1:
            notes1 = t1.read().strip()
        notes = (notes1,)

    for file in notes:
        root = ET.fromstring(file)
        root = root.findall("./docs")[0]
        for document in root.findall("./doc"):
            assert document.attrib['id'] not in documents
            documents[document.attrib['id']] = {}
            documents[document.attrib['id']]['text'] = document.findall("./text")[0].text

    annotation_files = tuple()
    if partition == 'train':
        with open('data/obesity_standoff_annotations_training.xml') as t1, \
                open('data/obesity_standoff_annotations_training_addendum.xml') as t2, \
                open('data/obesity_standoff_annotations_training_addendum2.xml') as t3, \
                open('data/obesity_standoff_annotations_training_addendum3.xml') as t4:
            train1 = t1.read().strip()
            train2 = t2.read().strip()
            train3 = t3.read().strip()
            train4 = t4.read().strip()
        # train1 = resource_string('clinical_data', 'phenotyping/n2c2_2008/train/obesity_standoff_annotations_training.xml').decode('utf-8').strip()
        # train2 = resource_string('clinical_data', 'phenotyping/n2c2_2008/train/obesity_standoff_annotations_training_addendum.xml').decode('utf-8').strip()
        # train3 = resource_string('clinical_data', 'phenotyping/n2c2_2008/train/obesity_standoff_annotations_training_addendum2.xml').decode('utf-8').strip()
        # train4 = resource_string('clinical_data', 'phenotyping/n2c2_2008/train/obesity_standoff_annotations_training_addendum3.xml').decode('utf-8').strip()
        annotation_files = (train1,train2,train3,train4)
    elif partition == 'test':
        with open('data/obesity_standoff_annotations_test.xml') as t1:
            test1 = t1.read().strip()
        # test1 = resource_string('clinical_data','phenotyping/n2c2_2008/test/obesity_standoff_annotations_test.xml').decode('utf-8').strip()
        annotation_files = (test1,)

    for file in annotation_files:
        root = ET.fromstring(file)
        for diseases_annotation in root.findall("./diseases"):

            annotation_source = diseases_annotation.attrib['source']
            assert isinstance(annotation_source, str)
            for disease in diseases_annotation.findall("./disease"):
                disease_name = disease.attrib['name']
                all_diseases.add(disease_name)
                for annotation in disease.findall("./doc"):
                    doc_id = annotation.attrib['id']
                    if not annotation_source in documents[doc_id]:
                        documents[doc_id][annotation_source] = {}
                    assert doc_id in documents
                    judgment = annotation.attrib['judgment']
                    documents[doc_id][annotation_source][disease_name] = judgment

    all_diseases = list(all_diseases)
    #print(all_diseases)

    for id in documents: #set unlabeled instances to None
        for annotation_type in ('textual', 'intuitive'):
            for disease in all_diseases:
                if not annotation_type in documents[id]:
                    documents[id][annotation_type] = {}
                if not disease in documents[id][annotation_type]:
                    #print(id, annotation_type, disease)
                    documents[id][annotation_type][disease] = None

    for id in documents:
        yield id, documents[id]['text'], documents[id]['textual'], documents[id]['intuitive']
    from pprint import pprint
    #pprint(documents[list(documents.keys())[1]])



