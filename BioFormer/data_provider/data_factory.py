from data_provider.uea import collate_fn,collate_fn_dep
from torch.utils.data import DataLoader
from data_provider.data_loader import (
    APAVALoader,
    ADFTDLoader,
    BCI2aLoader,
    BCI2bLoader,
    PTBLoader,
    PTBXLLoader,
    sub_Single_label_DependentLoader,
    sub_Multi_label_DependentLoader,
)


data_dict = {
    # Subject-Dependent setup
    "ADFTD-Dependent": sub_Single_label_DependentLoader,  # dataset ADFTD with subject-dependent setup
    "APAVA-Dependent": sub_Single_label_DependentLoader,  # dataset ADFTD with subject-dependent setup
    "PTB-Dependent": sub_Single_label_DependentLoader,  # dataset ADFTD with subject-dependent setup
    "PTB-XL-Dependent": sub_Single_label_DependentLoader,  # dataset ADFTD with subject-dependent setup
    "BCI2a-Dependent": sub_Multi_label_DependentLoader,  # dataset BCI2a with subject-dependent setup
    "BCI2b-Dependent": sub_Multi_label_DependentLoader,  #

    # Cross-Subject setup
    "APAVA": APAVALoader,       # dataset APAVA
    "BCI2a": BCI2aLoader, 
    "BCI2b": BCI2bLoader,  
    "ADFTD": ADFTDLoader,       # dataset ADFTD
    "PTB": PTBLoader,           # dataset PTB
    "PTB-XL": PTBXLLoader,      # dataset PTB-XL
}

def data_provider(args, flag):
    Data = data_dict[args.data]     # get the class of the dataset loader

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        if args.task_name == "classification":
            batch_size = args.batch_size
        else:
            batch_size = 1          # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == "classification":
        drop_last = False
        # create dataset
        data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )
        if args.is_cross_subject:
            # create across_subject dataloader
            data_loader = DataLoader(
                data_set,                               # dataset
                batch_size=batch_size,                  # batch size
                shuffle=shuffle_flag,                   # shuffle the data
                num_workers=args.num_workers,           # number of workers
                drop_last=drop_last,                    # drop the last batch if it's smaller than batch size
                collate_fn=lambda x: collate_fn(        # Defines how DataLoader concatenates multiple samples into a single batch
                    x, max_len=args.seq_len             # only called when yeilding batches
                ),  
            )
            return data_set, data_loader
        else:
            # create subject_dependet dataloader
            data_loader = DataLoader(
                data_set,                               # dataset
                batch_size=batch_size,                  # batch size
                shuffle=shuffle_flag,                   # shuffle the data
                num_workers=args.num_workers,           # number of workers
                drop_last=drop_last,                    # drop the last batch if it's smaller than batch size
                collate_fn=lambda x: collate_fn_dep(    # Defines how DataLoader concatenates multiple samples into a single batch
                    x, max_len=args.seq_len             # only called when yeilding batches
                ),  
            )
            return data_set, data_loader
