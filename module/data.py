import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    @staticmethod
    def split_segs(hist):        
        segs = []
        for idx, ids in enumerate(hist):
            seg_len = len(ids)
            if not idx % 2:
                segs.extend([0 for _ in range(seg_len)])
            else:
                segs.extend([1 for _ in range(seg_len)])
        return segs


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if isinstance(self.data[idx]['hist'][0], int):
            hist = self.data[idx]['hist']
            segs = [0 for _ in range(len(hist))]
        else:
            hist = []
            segs = self.split_segs(self.data[idx]['hist'])
            for x in self.data[idx]['hist']:
                hist.extend(x)

        uttr = self.data[idx]['uttr']
        resp = self.data[idx]['resp']
        
        return {'hist': hist, 'segs': segs,
                'uttr': uttr, 'resp': resp}



class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        hist_batch, segs_batch, uttr_batch, resp_batch = [], [], [], []
        
        for elem in batch:
            hist_batch.append(torch.LongTensor(elem['hist'])) 
            segs_batch.append(torch.LongTensor(elem['segs']))
            uttr_batch.append(torch.LongTensor(elem['uttr']))
            resp_batch.append(torch.LongTensor(elem['resp']))

        return {'hist': self.pad_batch(hist_batch),
                'segs': self.pad_batch(segs_batch),
                'uttr': self.pad_batch(uttr_batch),
                'resp': self.pad_batch(resp_batch)}

    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, split):
    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(config.pad_id), 
                      num_workers=2)
