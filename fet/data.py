from datasets import load_dataset
import os

def process_sentence(words, tags, neg_tag='O', labels=[]):
    entity_list = []
    start_index = 0

    for i in range(len(tags)):
        if i == 0 and tags[i] != neg_tag:
            start_index = i
            entity_type = labels[int(tags[i])]
        elif tags[i] != neg_tag and tags[i-1] == neg_tag:
            start_index = i
            entity_type = labels[int(tags[i])]

        elif tags[i] != neg_tag and tags[i-1] != neg_tag and tags[i-1] != tags[i]:
            end_index = i
            entity_name = ' '.join(words[start_index:end_index])
            if entity_type != "other":
                entity_list.append([entity_name, entity_type])
            start_index = i
            entity_type = labels[int(tags[i])]

        if i > 0:

            if tags[i] == neg_tag and tags[i-1] != neg_tag: #end of mention
                end_index = i
                entity_name = ' '.join(words[start_index:end_index])
                if entity_type != "other":
                    entity_list.append([entity_name, entity_type])
            elif i == len(tags) - 1 and tags[i] != neg_tag: # end of mention at eos
                end_index = i+1
                entity_name = ' '.join(words[start_index:end_index])
                entity_type = labels[int(tags[i])]
                if entity_type != "other":
                    entity_list.append([entity_name, entity_type])

    return list(zip([' '.join(words)]*len(entity_list), entity_list))