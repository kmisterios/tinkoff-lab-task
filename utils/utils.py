import numpy as np
from deep_translator import GoogleTranslator
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def rephrase(sentence):
    languages = ["de", "es", "ru", "fr", "zh-TW"]
    random_language = np.random.choice(languages)
    translator1 = GoogleTranslator(source='en', target=random_language)
    translator2 = GoogleTranslator(source=random_language, target="en")
    return translator2.translate(translator1.translate(sentence))


def augment_data(data, key, proportion):
    n = len(data["idx"])
    ids = [i for i in range(n)]
    random_ids = np.random.choice(ids, size = int(n * proportion))
    if key == "cola":
        data_for_translation = np.array(data["sentence"])[random_ids]
    else:
        data_for_translation = np.array([data["sentence1"], data["sentence2"]]).T[random_ids]
    for sentence, i in tqdm(zip(data_for_translation, random_ids), desc= f"Augmentation for {key}"):
        if key == "cola":
            data["sentence"].append(rephrase(sentence))
            #translated sentences always have the correct structure
            data["label"].append(1)
        else:
            data["sentence1"].append(rephrase(sentence[0]))
            data["sentence2"].append(rephrase(sentence[1]))
            data["label"].append(data["label"][i])
        data["idx"].append(data["idx"][-1] + 1)
    return data


def prepare_data(datasets, subset_size = 1.0, mode = "singleHead", augmentation = False, augmentation_proportion = 0.25):
    mode_types = ["singleHead", "multiHead", "singleTask"]
    assert mode in mode_types, f"mode should be one of {mode_types}"
    tags = {
        "cola" : "acceptability: ",
        "sst2" : "sentiment: ",
        "rte" : "nli: "
    }
    new_dict = {}
    datasets_lengths = {label : {part : len(datasets[label][part]) for part in datasets[label]} for label in datasets}
    for key in datasets:
        new_dict[key] = {}
        for df_part_label in datasets[key]:
            new_dict[key][df_part_label] = {}
            subset_len = int(subset_size * datasets_lengths[key][df_part_label])
            df_part = datasets[key][df_part_label][:subset_len]
            if augmentation and df_part_label == "train" and key != "sst2":
                df_part = augment_data(df_part, key, augmentation_proportion)
            new_dict[key][df_part_label]["idx"] = df_part["idx"]
            new_dict[key][df_part_label]["label"] = df_part["label"]
            if key != "rte" :
                sents = list(map(lambda x: tags[key] + x, df_part["sentence"])) \
                        if mode == "singleHead" else df_part["sentence"]
                new_dict[key][df_part_label]["sentence"] = sents
            else:
                thing_to_add = tags[key] if mode == "singleHead" else ""
                sents = [(sent1, sent2) for sent1, sent2 in zip(df_part["sentence1"], df_part["sentence2"])]
                updated_sents = list(map(lambda x: thing_to_add + x[0] + " [SEP] " + x[1], sents))
                new_dict[key][df_part_label]["sentence"] = updated_sents
    return new_dict


def create_dataloaders(data, batch_sizes = [8, 8, 8], model_name = "bert-base-uncased"):
    dataloaders = {key : [] for key in data["cola"]}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for key in data:
        for (data_part_label, data_part), batch_size in zip(data[key].items(), batch_sizes):
            labels_data  = [[1.,0.] if x == 1 else [0.,1.] for x in data_part["label"]]
            labels = torch.tensor(labels_data).float()
            tokens_info = tokenizer(data_part["sentence"], padding=True, truncation=True, return_tensors="pt")
            dataset_t = TensorDataset(tokens_info["input_ids"], tokens_info["attention_mask"], labels)
            sampler = RandomSampler(dataset_t)
            dataloaders[data_part_label].append(DataLoader(dataset_t, sampler=sampler, batch_size=batch_size))
    return dataloaders