import json


def main():
    ann = json.load(open('/data/ALBEF/data/dataset_flickr30k.json', 'r'))

    test_split = []
    count = 0
    for an in ann['images']:
        if an['split'] == 'test':
            if count >= 100:
                break
            test_split.append(an)
            count += 1

    json.dump({'images': test_split,
               'dataset': 'flickr30k'},
              open('/data/ALBEF/data/dataset_flickr30k_100.json', 'w+'))


if __name__ == '__main__':
    main()