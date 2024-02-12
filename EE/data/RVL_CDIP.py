import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pytesseract
from collections import OrderedDict

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D


def empty_image(height=2, width=2):
    i = np.ones((height, width, 3), np.uint8) * 255  # whitepage
    return i


def normalize_box(box, width, height):
    ## nasty fixes for wrong easyOCR bboxes
    bbox = [
        min(max(0, int(1000 * (box[0] / width))), 1000),
        min(max(0, int(1000 * (box[1] / height))), 1000),
        min(max(0, int(1000 * (box[2] / width))), 1000),
        min(max(0, int(1000 * (box[3] / height))), 1000),
    ]
    # assert all(0 <= coord <= 1000 for coord in bbox)

    return bbox


def apply_tessocr(image):
    width, height = image.size

    # apply ocr to the image
    ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int, errors="ignore")
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    # get the words and actual (unnormalized) bounding boxes
    words = [str(word) for word in ocr_df.text]
    return words


def process_single(example, processor):
    image = example["image"].convert("RGB")

    width, height = image.size

    # apply ocr to the image
    ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int, errors="ignore")
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    # get the words and actual (unnormalized) bounding boxes
    words = list(ocr_df.text)
    coordinates = ocr_df[["left", "top", "width", "height"]]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [
            x,
            y,
            x + w,
            y + h,
        ]  # we turn it into (left, top, left+widght, top+height) to get the actual box
        actual_boxes.append(actual_box)

    ## TODO: collect block bboxes [block_num]

    # normalize the bounding boxes
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))

    assert len(words) == len(boxes)

    # convert to token-level features
    encoding = convert_example_to_features(
        image, words, boxes, actual_boxes, processor.tokenizer
    )
    encoding["pixel_values"] = processor.feature_extractor(image)["pixel_values"][0]
    encoding["labels"] = torch.tensor(example["label"], dtype=torch.int32)
    return encoding


def convert_example_to_features(
    image,
    words,
    boxes,
    actual_boxes,
    tokenizer,
    max_seq_length=512,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
):
    width, height = image.size

    tokens = []
    token_boxes = []
    actual_bboxes = []  # we use an extra b because actual_boxes is already used
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        if isinstance(word, float):
            word = str(int(word))
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[
            : (max_seq_length - special_tokens_count)
        ]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]

    segment_ids = [0] * len(tokens)

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_boxes) == max_seq_length
    assert len(token_actual_boxes) == max_seq_length

    encoding = {}

    encoding["input_ids"] = torch.tensor(input_ids)
    encoding["bbox"] = torch.tensor(token_boxes)
    encoding["attention_mask"] = torch.tensor(input_mask)
    # encoding["token_type_ids"] = torch.tensor(segment_ids)
    return encoding


class RVL_CDIP(Dataset):
    """RVL-CDIP dataset (small subset)."""

    id2label = OrderedDict(
        {
            0: "letter",
            1: "form",
            2: "email",
            3: "handwritten",
            4: "advertisement",
            5: "scientific_report",
            6: "scientific_publication",
            7: "specification",
            8: "file_folder",
            9: "news_article",
            10: "budget",
            11: "invoice",
            12: "presentation",
            13: "questionnaire",
            14: "resume",
            15: "memo",
        }
    )
    label2id = OrderedDict({v: k for k, v in id2label.items()})

    def __init__(
        self,
        data,
        split,
        use_images=True,
        get_raw_ocr_data=True,
        processor=None,
        forward_signature=None,
    ):
        self.data = data
        self.split = split
        self.use_images = use_images
        self.get_raw_ocr_data = get_raw_ocr_data
        self.preprocessed = False
        self.forward_signature = forward_signature
        if processor is not None:
            self.processor = processor
            self.preprocess()  # encoded

    def __len__(self):
        return len(self.data)

    @property
    def num_labels(self):
        return len(type(self).id2label)

    def _features_(self):
        d = {}
        d["labels"] = ClassLabel(
            num_classes=self.num_labels, names=list(self.label2id.keys())
        )
        if self.use_images:
            image_column = (
                "pixel_values" if "pixel_values" in self.forward_signature else "image"
            )
            d[image_column] = Array3D(dtype="float32", shape=(3, 224, 224))

        if self.get_raw_ocr_data:
            d["input_ids"] = Sequence(feature=Value(dtype="int64"))
            d["attention_mask"] = Sequence(Value(dtype="int64"))

            # "token_type_ids": Sequence(Value(dtype="int64")),

        if self.use_images and self.get_raw_ocr_data:
            d["bbox"] = Array2D(dtype="int64", shape=(512, 4))

        # if sorted(d.keys()) != sorted(self.forward_signature):

        self.features = Features(d)
        return self.features

    def preprocess_data(self, examples):
        # take a batch of images
        if self.use_images:
            raw = [image.convert("RGB") for image in examples["image"]]

        if self.get_raw_ocr_data and not self.use_images:
            raw = [
                " ".join(apply_tessocr(image.convert("RGB")))
                for image in examples["image"]
            ]

        encoded_inputs = self.processor(
            raw, padding="max_length", truncation=True
        )  # does OCR and all the rest

        # add labels
        label_column = "label" if "label" in examples else "labels"
        if isinstance(examples[label_column], str):
            encoded_inputs["labels"] = [
                self.label2id[label] for label in examples[label_column]
            ]
        else:
            encoded_inputs["labels"] = examples[label_column]

        if "token_type_ids" in encoded_inputs:
            encoded_inputs.pop("token_type_ids")
        """
        for k in encoded_inputs:
            if k not in self.forward_signature:
                print(f"unsupported column: {k}")
        """
        return encoded_inputs

    def preprocess(self):
        # we need to define custom features
        features = self._features_()

        # DEV: multiprocess edits -> stupid, have to re-cache
        num_proc = 1
        batch_size = 10

        # with OCR if passed as model config
        self.data = self.data.map(
            self.preprocess_data,
            remove_columns=[
                col
                for col in self.data.column_names
                if col not in self.forward_signature
            ],
            features=features,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
        )

        # with manual OCR
        # self.data = self.data.map(
        #     lambda ex: process_single(ex, self.processor),
        #     desc="precomputing features",
        #     keep_in_memory=False,
        #     remove_columns=["image", "label"],
        # )
        self.data.set_format("torch")
        self.preprocessed = True

    def sample(self, idx=None):
        if idx is not None:
            return self.__getitem__(idx)
        idx = random.randint(0, self.__len__())
        return self.__getitem__(idx)

    def __getitem__(self, idx, processor=None):
        if self.preprocessed:
            return self.data[idx]
        else:
            if hasattr(self, "processor"):
                return self.preprocess_data(self.data[idx])
            return self.data[idx]


class RVL_CDIP_IO(RVL_CDIP):
    """RVL-CDIP dataset from EasyOCR."""

    def __init__(
        self,
        data,
        split,
        use_images=True,
        get_raw_ocr_data=True,
        processor=None,
        forward_signature=None,
    ):
        super().__init__(
            data=data,
            split=split,
            use_images=use_images,
            get_raw_ocr_data=get_raw_ocr_data,
            forward_signature=forward_signature,
            processor=None,
        )
        if processor is not None:
            self.processor = processor
            self.processor.image_processor.apply_ocr = False
            self.preprocess()  # encoded

    def preprocess_data(self, examples):
        # take a batch of images
        if self.use_images:
            raw = [image.convert("RGB") for image in examples["image"]]
            boxes = [
                [normalize_box(box, *raw[i].size) for box in bbox]
                for i, bbox in enumerate(examples["boxes"])
            ]
            encoded_inputs = self.processor(
                raw,
                text=examples["words"],
                boxes=boxes,
                padding="max_length",
                truncation=True,
                # split_on_spaces=False
                # is_split_into_words=True,
            )

        if self.get_raw_ocr_data and not self.use_images:
            # raw = [" ".join(words) for words in examples["words"]]
            encoded_inputs = self.processor.tokenizer(
                examples["words"],
                boxes=examples["boxes"],  # impossible without image
                padding="max_length",
                truncation=True,
            )

        # add labels
        label_column = "label" if "label" in examples else "labels"
        if isinstance(examples[label_column], str):
            encoded_inputs["labels"] = [
                self.label2id[label] for label in examples[label_column]
            ]
        else:
            encoded_inputs["labels"] = examples[label_column]

        if "token_type_ids" in encoded_inputs:
            encoded_inputs.pop("token_type_ids")

        return encoded_inputs

    def preprocess(self):
        self.data = self.data.map(
            self.preprocess_data,
            remove_columns=[
                col
                for col in self.data.column_names
                if col not in self.forward_signature
            ],
            features=self._features_(),
            batched=True,
            batch_size=50,
            num_proc=40,
        )
        self.data.set_format("torch")
        self.preprocessed = True


class Tobacco3482(RVL_CDIP):
    """Tobacco-3482 dataset"""

    id2label = OrderedDict(
        {
            0: "ADVE",
            1: "Email",
            2: "Form",
            3: "Letter",
            4: "Memo",
            5: "News",
            6: "Note",
            7: "Report",
            8: "Resume",
            9: "Scientific",
        }
    )
    label2id = OrderedDict({v: k for k, v in id2label.items()})

    def __init__(
        self,
        data,
        split,
        use_images=True,
        get_raw_ocr_data=True,
        processor=None,
        forward_signature=None,
    ):
        super().__init__(
            data=data,
            split=split,
            use_images=use_images,
            get_raw_ocr_data=get_raw_ocr_data,
            forward_signature=forward_signature,
            processor=processor,
        )


def create_new_rvl():
    from copy import deepcopy
    from tqdm import tqdm
    from datasets import load_dataset

    dataset = load_dataset("rvl_cdip", cache_dir="/mnt/lerna/data/HFcache")

    def split_get_samples(datasplit, N_K=25):
        # shuffle first
        datasplit = datasplit.shuffle(
            seed=42, keep_in_memory=True
        )  # maybe this might be the issue?

        idx = {k: [] for k in RVL_CDIP.id2label.keys()}
        for i, x in enumerate(datasplit):
            if len(idx[x["label"]]) < N_K:
                idx[x["label"]].append(i)
            if all([len(idx[k]) == N_K for k, v in idx.items()]):
                break
        all_indices = []
        for k, v in idx.items():
            all_indices.extend(v)
        datasplit = datasplit.select(all_indices)
        print(f"After filtering: {len(datasplit)}")

        return datasplit

    dataset["train"] = split_get_samples(dataset["train"], N_K=50)
    dataset["validation"] = split_get_samples(dataset["validation"], N_K=25)
    dataset["test"] = split_get_samples(dataset["test"], N_K=25)
    dataset.push_to_hub("jordyvl/rvl_cdip_100_examples_per_class")


if __name__ == "__main__":
    create_new_rvl()
