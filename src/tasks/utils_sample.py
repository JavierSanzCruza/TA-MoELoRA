import json
import random
from collections import defaultdict
from tqdm import tqdm

random.seed(42)
negative_ratio=1.0


def filter_sentence(json_data, key, max_words_num=3):
    new_json = []
    for entry in json_data:
        words = entry[key].split(' ')
        if len(words)>max_words_num:
            new_json.append(entry)
    print('before filtering:', len(json_data))
    print('after filtering:', len(new_json))
    return new_json

### or token list is provided 
def filter_tokens(json_data, key, max_words_num=3):
    new_json = []
    for entry in json_data:
        words = entry[key]
        if len(words)>max_words_num:
            new_json.append(entry)
    print('before filtering:', len(json_data))
    print('after filtering:', len(new_json))
    return new_json

def sample_unlabelld(json_data, mention_key, type_key):
    labeled_data = []
    unlabeled_data = []
    
    for entry in json_data:
        if len(entry[mention_key])>0:
            labeled_data.append(entry)
        else:
            unlabeled_data.append(entry)
    sampled_labeled_data, label_counts = sample_per_type(labeled_data, mention_key, type_key)
    num_labeled = len(sampled_labeled_data)
    num_unlabelled = len(unlabeled_data)
    print('labelled data nums:', num_labeled, 'unlabelled data nums:', num_unlabelled)
    
    if num_labeled < num_unlabelled:
        
        num_unlabeled_to_sample = int(num_labeled * negative_ratio)
        num_unlabeled_to_sample = min(num_unlabeled_to_sample, len(unlabeled_data))
        
        sampled_unlabeled_data = random.sample(unlabeled_data, num_unlabeled_to_sample)
        combined_data = sampled_labeled_data + sampled_unlabeled_data
        random.shuffle(combined_data)
        print('total sampled amount: ', len(combined_data))
        return combined_data, label_counts
    else:
        combined_data = sampled_labeled_data + unlabeled_data
        random.shuffle(combined_data)
        print('total sampled amount: ', len(combined_data))
        return combined_data, label_counts

def sample_per_type(json_data, mention_key, type_key, max_mentions_per_type = 500):
    
    random.shuffle(json_data)
    all_types = set()
    for item in json_data:
        for mention in item.get(mention_key, []):
            if type_key == 'typed':
                mention_type = type(mention)
            else:
                mention_type = mention.get(type_key)
            all_types.add(mention_type)
    total_types = len(all_types)

    # init counters
    type_counts = defaultdict(int)
    output_data = []
    types_reached_limit = set()

    for item in tqdm(json_data):
        # get all annotations of the given sentence example
        types_in_item = set()
        for mention in item.get(mention_key, []):
            if type_key == 'typed':
                mention_type = type(mention)
            else:
                mention_type = mention.get(type_key)
            types_in_item.add(mention_type)
    
        should_add = False
        for mention_type in types_in_item:
            if type_counts[mention_type] < max_mentions_per_type:
                should_add = True
                break
    
        if should_add:
            output_data.append(item)
            for mention_type in types_in_item:
                if type_counts[mention_type] < max_mentions_per_type:
                    type_counts[mention_type] += 1
                    if type_counts[mention_type] == max_mentions_per_type:
                        types_reached_limit.add(mention_type)
    
        if len(types_reached_limit) == total_types:
            break
    
    return output_data, type_counts

