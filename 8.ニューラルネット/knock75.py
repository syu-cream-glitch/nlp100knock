import os
import torch

def collate(batch):
    lengths = [len(x['input_ids']) for x in batch]

    sorted_idxs = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
    sorted_batch = [batch[i] for i in sorted_idxs]

    max_length = max(lengths)
    input_ids_padded = []
    labels = []

    for data in sorted_batch:
        input_ids = data['input_ids']
        pad_length = max_length - len(input_ids)
        padding = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
        input_ids_padded.append(padding)
        labels.append(data['label'])
    
    input_ids_tensor = torch.stack(input_ids_padded)
    labels_tensor = torch.stack(labels)

    return {'input_ids': input_ids_tensor, 'label': labels_tensor}

# 動作確認
example = [
    {
        'text': 'hide new secretions from the parental units',
        'label': torch.tensor([0.]),
        'input_ids': torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594])
    },
    {
        'text': 'contains no wit , only labored gags',
        'label': torch.tensor([0.]),
        'input_ids': torch.tensor([3475, 87, 15888, 90, 27695, 42637])
    },
    {
        'text': 'that loves its characters and communicates something rather beautiful about human nature',
        'label': torch.tensor([1.]),
        'input_ids': torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964])
    },
    {
        'text': 'remains utterly satisfied to remain the same throughout',
        'label': torch.tensor([0.]),
        'input_ids': torch.tensor([987, 14528, 4941, 873, 12, 208, 898])
    }
]

collated_batch = collate(example)

os.makedirs('output', exist_ok=True)
with open('output/output75.txt', 'w', encoding='utf-8') as f:
    f.write(str(collated_batch))