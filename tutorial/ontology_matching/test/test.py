import json
import os
import xml.dom.minidom as xml


def align_parser(align_file):
    '''
    :param align_file: ground truth
    :return: (entity1, entity2, measure, relation) ...
    '''
    pairs = []
    dom = xml.parse(align_file)
    maps = dom.getElementsByTagName('map')
    for item in maps:
        entity1 = item.getElementsByTagName('entity1')[0].getAttribute('rdf:resource')
        entity2 = item.getElementsByTagName('entity2')[0].getAttribute('rdf:resource')
        measure = item.getElementsByTagName('measure')[0].firstChild.data
        relation = item.getElementsByTagName('relation')[0].firstChild.data
        pairs.append((entity1, entity2, measure, relation))

    return pairs


def main():
    data_dir = '../datasets'
    out_dir = '../output'

    align_list = ['301', '302', '303', '304']

    for res_file in os.listdir(out_dir):
        # check the file name
        file_split = res_file.split('.')[0].split('-')
        if len(file_split) != 2 or file_split[0] != '101' or file_split[1] not in align_list:
            print('{} is not a correct file name!'.format(res_file))
            continue

        print('----------- {} -----------'.format(file_split[1]))
        # ground truth
        align_file = os.path.join(os.path.join(data_dir, file_split[1]), 'refalign.rdf')
        ground = align_parser(align_file)

        # matching results
        with open(os.path.join(out_dir, res_file), 'r') as f:
            pred = json.load(f)

        # calculate precision and recall
        len_pred = len(pred)    # TP + FP
        len_true = len(ground)  # TP + FN
        tp = 0                  # TP
        for i in range(len_pred):
            for j in range(len(ground)):
                # ignore "measure"
                if pred[i][0]==ground[j][0] and pred[i][1]==ground[j][1] and pred[i][3]==ground[j][3]:
                    tp += 1
                    ground.pop(j)
                    break

        precision = tp/len_pred
        recall = tp/len_true
        f1 = 2*precision*recall/(precision+recall)
        print('Precision: {}/{} = {}'.format(tp, len_pred, precision))
        print('Recall: {}/{} = {}'.format(tp, len_true, recall))
        print('F1: {}'.format(f1))


if __name__ == '__main__':
    main()
