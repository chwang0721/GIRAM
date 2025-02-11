import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config_GETNext import args
from dataset import load_data, dataloader, pretrain_dataloader
from logging_set import get_logger
from trainers.Trainer_GETNext import Trainer_GETNext
from trainers.utils import load_graph_adj_mtx, load_graph_node_features, calculate_laplacian_matrix


def cal_A_X():
    raw_A = load_graph_adj_mtx(f'data/{args.dataset}/graph_A.csv')
    raw_X = load_graph_node_features(f'data/{args.dataset}/graph_X.csv')

    raw_A = torch.from_numpy(raw_A).to(args.device, dtype=torch.float)
    A = calculate_laplacian_matrix(raw_A)

    raw_X = torch.from_numpy(raw_X).to(args.device, dtype=torch.float)
    cat_list = raw_X[:, 1].long()
    num_cats = cat_list.max().item() + 1
    one_hot_rlt = F.one_hot(cat_list, num_classes=num_cats).float()
    X = torch.zeros(
        (raw_X.shape[0], raw_X.shape[1] - 1 + num_cats),
        dtype=torch.float32,
        device=args.device
    )

    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]

    in_channels = raw_X.shape[-1] - 1 + num_cats
    return A, X, in_channels


if __name__ == '__main__':
    dataset = args.dataset
    batch_size = args.batch_size
    mode = args.mode

    df_all = pd.read_csv(f'data/{dataset}/all.csv')
    poi_num = df_all['PoiId'].nunique()
    user_num = df_all['UserId'].nunique()
    main_cat_num = df_all['PoiMainCatId'].nunique()
    cat_num = df_all['PoiCatId'].nunique()
    min_lat, max_lat = df_all['Latitude'].min(), df_all['Latitude'].max()
    min_lon, max_lon = df_all['Longitude'].min(), df_all['Longitude'].max()
    geo_range = ((min_lat, max_lat), (min_lon, max_lon))
    region_num = df_all['GridId'].nunique()
    del df_all

    if dataset == 'NYC':
        memory_size = 100
    elif dataset == 'TKY':
        memory_size = 100
    elif dataset == 'Gowalla':
        memory_size = 20

    block_num = 5

    log_path = f'logs/{dataset}/GETNext'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = get_logger(log_path + f'/{mode}.log')

    cpt_path = f'checkpoints/{dataset}/GETNext'
    if not os.path.exists(cpt_path):
        os.makedirs(cpt_path)
    pretrain_path = cpt_path + '/pretrain.pt'
    pretrain_path_nu = cpt_path + '/pretrain_nu.pt'

    train_dataset_0 = load_data(dataset, 0)
    test_dataset_0 = load_data(dataset, 1)
    train_loader_0, test_loader_0 = pretrain_dataloader(train_dataset_0, test_dataset_0, batch_size)
    train_dataset_0 = load_data(dataset, 0)

    logger.info('========================================')
    logger.info(f'Dataset: {dataset}')

    A, X, in_channels = cal_A_X()

    trainer = Trainer_GETNext(args, poi_num, user_num, cat_num, main_cat_num, memory_size, A, X, in_channels, geo_range,
                              region_num)

    if mode == 'pretrain':
        last_acc5 = 0
        last_acc5_nu = 0
        for epoch in range(args.base_epochs):
            train_loss = trainer.train(train_loader_0)
            train_loss_nu = trainer.train_nu(train_loader_0)
            logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Loss_nu: {train_loss_nu:.4f}')
            acc5, acc10, acc20, mrr, acc5_nu, acc10_nu, acc20_nu, mrr_nu = trainer.test_pretrain(test_loader_0)
            if acc5 > last_acc5:
                last_acc5 = acc5
                torch.save(trainer.model.state_dict(), pretrain_path)
            if acc5_nu > last_acc5_nu:
                last_acc5_nu = acc5_nu
                torch.save(trainer.model_nu.state_dict(), pretrain_path_nu)

    else:
        logger.info(f'Mode: {mode}')
        if mode != 'retrain':
            trainer.model.load_state_dict(torch.load(pretrain_path, weights_only=True))
        if mode == 'memory':
            trainer.model_nu.load_state_dict(torch.load(pretrain_path_nu, weights_only=True))

        accs5, accs10, accs20, mrrs = [], [], [], []
        if mode == 'finetune':
            for idx in range(1, block_num):
                last_acc5 = 0
                train_dataset = load_data(dataset, idx)
                next_dataset = load_data(dataset, idx + 1)
                train_loader, valid_loader, test_loader = dataloader(train_dataset, next_dataset, batch_size)

                for epoch in range(args.incremental_epochs):
                    train_loss = trainer.train(train_loader)
                    acc5, acc10, acc20, mrr = trainer.test(valid_loader)
                    if acc5 > last_acc5:
                        last_acc5 = acc5
                        torch.save(trainer.model.state_dict(), cpt_path + f'/finetune_{idx}.pt')

                trainer.model.load_state_dict(torch.load(cpt_path + f'/finetune_{idx}.pt', weights_only=True))
                acc5, acc10, acc20, mrr = trainer.test(test_loader)
                logger.info(f'Acc@5: {acc5:.4f}, Acc@10: {acc10:.4f}, Acc@20: {acc20:.4f}, MRR: {mrr:.4f}')

                accs5.append(acc5)
                accs10.append(acc10)
                accs20.append(acc20)
                mrrs.append(mrr)

        elif mode == 'retrain':
            for idx in range(1, block_num):
                last_acc5 = 0
                trainer = Trainer_GETNext(args, poi_num, user_num, cat_num, main_cat_num,
                                          memory_size, A, X, in_channels, geo_range, region_num)
                train_dataset = load_data(dataset, idx)
                next_dataset = load_data(dataset, idx + 1)

                train_dataset = torch.utils.data.ConcatDataset([train_dataset_0, train_dataset])
                train_loader, valid_loader, test_loader = dataloader(train_dataset, next_dataset, batch_size)
                train_dataset_0 = train_dataset

                for epoch in range(args.incremental_epochs):
                    train_loss = trainer.train(train_loader)
                    acc5, acc10, acc20, mrr = trainer.test(valid_loader)
                    if acc5 > last_acc5:
                        last_acc5 = acc5
                        torch.save(trainer.model.state_dict(), cpt_path + f'/retrain_{idx}.pt')

                trainer.model.load_state_dict(torch.load(cpt_path + f'/retrain_{idx}.pt', weights_only=True))
                acc5, acc10, acc20, mrr = trainer.test(test_loader)
                logger.info(f'Acc@5: {acc5:.4f}, Acc@10: {acc10:.4f}, Acc@20: {acc20:.4f}, MRR: {mrr:.4f}')

                accs5.append(acc5)
                accs10.append(acc10)
                accs20.append(acc20)
                mrrs.append(mrr)

        elif mode == 'memory':
            accs1, accs5, accs10, accs20, mrrs = [], [], [], [], []
            user_sim = trainer.get_similarities(train_loader_0)
            trainer.update_memory(train_loader_0, user_sim)
            trainer.train_vae()

            for idx in range(1, block_num):
                last_acc5 = 0
                last_acc5_nu = 0

                train_dataset = load_data(dataset, idx)
                next_dataset = load_data(dataset, idx + 1)
                train_loader, valid_loader, test_loader = dataloader(train_dataset, next_dataset, batch_size)

                for epoch in range(args.incremental_epochs):
                    train_loss = trainer.train(train_loader)
                    train_loss_nu = trainer.train_nu(train_loader)
                    acc5, acc10, acc20, mrr, acc5_nu, acc10_nu, acc20_nu, mrr_nu = trainer.test_pretrain(valid_loader)
                    if acc5 > last_acc5:
                        last_acc5 = acc5
                        torch.save(trainer.model.state_dict(), cpt_path + f'/memory_{idx}.pt')
                    if acc5_nu > last_acc5_nu:
                        last_acc5_nu = acc5_nu
                        torch.save(trainer.model_nu.state_dict(), cpt_path + f'/memory_nu_{idx}.pt')

                trainer.model.load_state_dict(torch.load(cpt_path + f'/memory_{idx}.pt', weights_only=True))
                trainer.model_nu.load_state_dict(torch.load(cpt_path + f'/memory_nu_{idx}.pt', weights_only=True))

                user_sim = trainer.get_similarities(train_loader)
                trainer.update_memory(train_loader, user_sim)
                acc5, acc10, acc20, mrr = trainer.test_memory(test_loader, user_sim)
                logger.info(f'Acc@5: {acc5:.4f}, Acc@10: {acc10:.4f}, Acc@20: {acc20:.4f}, MRR: {mrr:.4f}')

                accs5.append(acc5)
                accs10.append(acc10)
                accs20.append(acc20)
                mrrs.append(mrr)

        logger.info('========================================')
        logger.info('Final Results:')
        logger.info(f'Acc@5: {np.mean(accs5):.4f}, Acc@10: {np.mean(accs10):.4f}, '
                    f'Acc@20: {np.mean(accs20):.4f}, MRR: {np.mean(mrrs):.4f}')
        logger.info('========================================')
