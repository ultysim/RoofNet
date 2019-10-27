import numpy as np
import torch 
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image

def str_to_date(s):
    return datetime.strptime(s, '%m/%d/%Y')

def get_transition_year(date1,date2):
    eoy = datetime(date1.year,12,31,0,0)
    diff = (date2-date1).days
    delta1 = (eoy - date1).days
    delta2 = (eoy - date2).days
    if delta2 > 0:
        return eoy.year
    if delta1 > abs(delta2):
        return eoy.year
    else:
        return eoy.year + 1
    
class ImageDataset(Dataset):
    """
    Creates image dataset of 32X64 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, 
                file_path,
                transform=None):
        print('Loading data')
        data_dict = np.load(file_path, allow_pickle=True)
        data_dict = data_dict.item().get('data')

        self.data = []
        self.data_dict = data_dict

            
        for k,v in data_dict.items():
            if type(v['expiration']) == str and type(v['issue']) == str:
                pass
            else:
                continue
            m = len(v['years'])
            sorted_args = np.argsort(v['years'])
            sorted_v = {key:[] for key in v.keys()}
            for i in sorted_args:
                for key in sorted_v.keys():
                    if key != 'trans_year':
                        sorted_v[key].append(v[key][i])
                    else:
                        sorted_v[key].append(v[key])
                    
            for i in range(m):
                if v['trans_year'] != 0:
                    transition_year = v['trans_year']
                else:
                    issue = str_to_date(v['issue'])
                    expiration = str_to_date(v['expiration'])
                    transition_year = get_transition_year(issue,expiration)

                self.data.append((sorted_v['imgs'][i],sorted_v['years'][i],transition_year,v['address']))
            
        
        print('Done loading data')
 
        self.transform = transform

        self.length = len(self.data)
        
        self.num_roofs = len(list(self.data_dict.keys()))

        print('Length', self.length)
        print('Num Roofs', self.num_roofs)
        #assert self.length % self.num_roofs == 0
        self.roofs_per_group = self.length // self.num_roofs
        
        
        index = 0
        for key in data_dict.keys():
            for j in range(self.roofs_per_group):
                if j == 0:
                    self.data_dict[key]["index"] = []
                self.data_dict[key]["index"].append(index)
                index+=1
            
 
    def __getitem__(self, index): 
        img, year, transition_year,name = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, year,{'transition_year':transition_year,'address':name}

    def __len__(self):
        return self.length

    

class TripletBuildingSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self,
                 data_source,
                 batch_size=32,
                 ):

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_roofs = len(list(self.data_source.data_dict.keys()))
        
        assert len(data_source) % self.num_roofs == 0
        self.roofs_per_group = len(data_source) // self.num_roofs        

    def _one_batch(self):
        
        roof_id = np.random.randint(self.num_roofs)
        ids = range(roof_id*self.roofs_per_group,(1+roof_id)*self.roofs_per_group)
        data = [self.data_source[i] for i in ids]
        
        trans_year = data[0][2]["transition_year"]
        
        years = np.array([data[i][1] for i in range(len(data))])
        trans_id = (np.abs(years - trans_year)).argmin()
  
        if trans_id > 0:
            redundant = True
            while redundant:
            
                ref_id,pos_id = np.random.choice(range(trans_id+1),2,replace=False)
                neg_id = np.random.choice(range(trans_id,len(years)))
                if len(set([ref_id,pos_id,neg_id]))==3:
                    redundant=False
            
        else:
            ref_id,pos_id = np.random.choice(range(1,len(years)),2,replace=False)
            neg_id = 0
            
            
        n = roof_id * self.roofs_per_group
        ref_id+=n
        pos_id+=n
        neg_id+=n
        return [ref_id,pos_id,neg_id]

        
    def __iter__(self):

        minibatch = np.array([self._one_batch()
                              for _ in range(self.batch_size)]).reshape(-1)

        return iter(minibatch)

    def __len__(self):
        return len(self.data_source)
