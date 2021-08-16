import numpy as np
from sklearn.model_selection import train_test_split
import funcy
from tabulate import tabulate
import coloredlogs, logging
import itertools, os, json, urllib.request

coloredlogs.install()


def check_instances_categories(file, annotations, class_names):
    """
    #### category index should start from 1
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,))
    for anno in annotations:
        classes = np.asarray(
            [anno["category_id"] - 1]
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                    classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logging.basicConfig(format='[%(asctime)s : %(message)s %(filename)s]',
                        log_colors='green', loglevel=logging.INFO)

    logging.info('\n' + '\033[92m' + 'Categories and Instances in the ' + file + ':' + '\033[96m' + '\n' + table)


def save_coco(file, images, annotations, categories):
    check_instances_categories(file, annotations, [category['name'] for category in categories])
    with open(file, 'wt') as coco:
        json.dump({'images': images, 'annotations': annotations, 'categories': categories}, coco, indent=2,
                  sort_keys=False)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def dataset_split(annotation_file, path_to_train, path_to_test, ratio):
    with open(annotation_file, 'rt') as annotations:
        coco = json.load(annotations)
        images = coco['images']

        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        train, test = train_test_split(images, train_size=ratio)

        save_coco(path_to_train, train, filter_annotations(annotations, train), categories)
        save_coco(path_to_test, test, filter_annotations(annotations, test), categories)


def check_download_images(imgs_info):
    for img_info in imgs_info:
        image_path = img_info['image_path']
        image_url = img_info['url']
        f_path = os.path.abspath(os.path.dirname(image_path) + os.path.sep + ".")
        if os.access(image_path, mode=os.R_OK):
            continue
        else:
            os.makedirs(f_path, exist_ok=True)
            try:
                urllib.request.urlretrieve(image_url, image_path)
            except Exception as e:
                print("Error occurred when downloading image: {}, error message:".format(img_info['file_name']))
                print(e)


def check_anno_index(path_to_anno):
    with open(path_to_anno) as coco_format_anno:
        anno = json.load(coco_format_anno)
    annotations = anno['annotations']
    categories = anno['categories']
    index_start_zero = False
    if categories[0]['id'] != 0:
        return index_start_zero, anno
    else:
        index_start_zero = True
        for category in categories:
            category['id'] += 1
        for annotation in annotations:
            annotation['category_id'] += 1
    anno_sorted_index = {
        "images": anno['images'],
        "annotations": annotations,
        "categoreis": categories
    }
    return index_start_zero, anno_sorted_index
