
import torch.nn.utils as utils
import random
from parse import parse_args
import time
import tqdm
import CPLSTM_dataset
import control_tower
import warnings
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
import argparse

matplotlib.use('TkAgg')  # 或者 matplotlib.use('Agg')
warnings.filterwarnings(action='ignore')

from tslearn.metrics import dtw, dtw_path

args = parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def evaluate(epoch, model, optimizer, dataloader, model_name, times, loss_fn, NLL_fn):
    model.eval()
    total_dataset_size = 0
    total_mse = []
    total_dtw = []
    total_dt = []
    total_loss = []
    total_tdi = []

    for batch in dataloader:
        loss_tdi_f = []
        loss_dtw_f = []

        batch_coeffs, batch_x, batch_y = batch
        pred_y, dydt = model(args, batch_x, batch_coeffs, times)

        mse = loss_fn(pred_y, batch_y)
        batch_y_epi = batch_y[:, 1:, :]
        batch_y_pre = batch_y[:, :-1, :]
        batch_y_diff = batch_y_epi - batch_y_pre
        dt_loss = loss_fn(dydt, batch_y_diff)
        batch_size = batch_y.shape[0]
        features = batch_y.shape[-1]
        loss = (args.alpha * mse) + (args.beta * dt_loss)
        for f in range(features):
            loss_tdi_ = 0
            loss_dtw_ = 0

            for k in range(batch_size):
                target_k_cpu = batch_y[k, :, f].view(-1).detach().cpu().numpy()
                output_k_cpu = pred_y[k, :, f].view(-1).detach().cpu().numpy()

                path, sim = dtw_path(target_k_cpu, output_k_cpu)
                loss_dtw_ += sim

                Dist = 0
                for i, j in path:
                    Dist += (i - j) * (i - j)
                loss_tdi_ += Dist / (args.pred_len * args.pred_len)
            loss_tdi_f.append(loss_tdi_ / batch_size)
            loss_dtw_f.append(loss_dtw_ / batch_size)

        loss_dtw = np.average(loss_dtw_f)
        loss_tdi = np.average(loss_tdi_f)
        b_size = batch_y.size(0)

        total_dataset_size += b_size
        total_mse.append(mse.item())
        total_dt.append(dt_loss.item())
        total_dtw.append(loss_dtw)
        total_tdi.append(loss_tdi)
        total_loss.append(loss.item())
    total_mse = np.average(total_mse)
    total_dtw = np.average(total_dtw)
    total_dt = np.average(total_dt)
    total_loss = np.average(total_loss)
    total_tdi = np.average(total_tdi)
    return total_mse, total_dtw, total_dt, total_tdi, total_loss


def load_model(args, model_path, visualize_version='test'):
    device = "cuda"
    model_name = args.model
    print("开始测试预加载的模型")

    train_dataloader, val_dataloader, test_dataloader, input_channels, output_channels = CPLSTM_dataset.get_dataset(
        args, device, visualization=True)
    model = control_tower.Model_selection_part(args, input_channels=input_channels, output_channels=output_channels,
                                               device=device)

    times = torch.Tensor(np.arange(args.seq_len))
    model = model.to(device)
    times = times.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fn = torch.nn.MSELoss()
    NLL_fn = torch.nn.NLLLoss()
    ckpt_file = model_path

    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    breaking = False
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None
    for epoch in range(1):
        if breaking:
            break
        model.eval()
        total_dataset_size = 0
        full_pred_y = torch.Tensor()
        full_true_y = torch.Tensor()
        full_x = torch.Tensor()
        loss_dtw = []
        loss_tdi = []
        preds = []
        trues = []
        if visualize_version == 'train':
            dataloader = train_dataloader
        elif visualize_version == 'val':
            dataloader = val_dataloader
        else:
            dataloader = test_dataloader
        i=0
        for batch in dataloader:
            loss_tdi_f = []
            loss_dtw_f = []
            batch_coeffs, batch_x, batch_y = batch
            if breaking:
                break
            pred_y, pred_prob = model(args, batch_x, batch_coeffs, times)
            b_size = batch_y.size(0)
            mse = loss_fn(pred_y, batch_y)
            pred = pred_y.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            batch_y_epi = batch_y[:, 1:, :]
            batch_y_pre = batch_y[:, :-1, :]
            batch_y_diff = batch_y_epi - batch_y_pre
            true_prob = (batch_y_diff > 0).to(batch_y.dtype)
            pred_prob = pred_prob.to(batch_y.dtype)
            batch_size = batch_y.shape[0]
            features = batch_y.shape[-1]
            for f in range(features):
                loss_tdi_ = 0
                loss_dtw_ = 0
                for k in range(batch_size):
                    target_k_cpu = batch_y[k, :, f].view(-1).detach().cpu().numpy()
                    output_k_cpu = pred_y[k, :, f].view(-1).detach().cpu().numpy()

                    path, sim = dtw_path(target_k_cpu, output_k_cpu)
                    loss_dtw_ += sim

                    Dist = 0
                    for i, j in path:
                        Dist += (i - j) * (i - j)
                    loss_tdi_ += Dist / (args.pred_len * args.pred_len)
                loss_tdi_f.append(loss_tdi_ / batch_size)
                loss_dtw_f.append(loss_dtw_ / batch_size)

            loss_dtw = np.average(loss_dtw_f)
            loss_tdi = np.average(loss_tdi_f)
            preds.append(pred)
            trues.append(true)

            full_pred_y = torch.cat([full_pred_y, pred_y.squeeze(-1).cpu()], dim=0)
            full_true_y = torch.cat([full_true_y, batch_y.squeeze(-1).cpu()], dim=0)
            full_x = torch.cat([full_x, batch_x.cpu()], dim=0)
            optimizer.zero_grad()
            total_dataset_size += b_size

        preds = np.array(preds)

        trues = np.array(trues)
        # 绘制图形
        plt.figure(figsize=(15, 8))

        preds_selected = preds[0, :, 0, :]  # 第一个时间步，选择第一个样本和第一个特征
        trues_selected = trues[0, :, 0, :]  # 同样选择真实值

        trues_selected = CPLSTM_dataset.inverse_transform(args, trues_selected)
        preds_selected = CPLSTM_dataset.inverse_transform(args, preds_selected)

        preds_selected = preds_selected[:, 0]  # 第一个时间步，选择第一个样本和第一个特征
        trues_selected = trues_selected[:, 0]  # 同样选择真实值
        print(trues_selected)
        save_selected_data(preds_selected, trues_selected, args)
        # 绘制预测值和真实值的折线图
        plt.plot(preds_selected, label='Predictions', color='b', linestyle='-', marker='o')
        plt.plot(trues_selected, label='True Values', color='r', linestyle='--', marker='x')
        # 添加标题和标签
        plt.title('Predictions vs True Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        # 显示图例
        plt.legend()
        # 显示网格
        plt.grid(True)
        # 显示图形
        plt.show()
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        return mae, mse, rmse, mape, mspe, rse, corr, np.average(loss_tdi), np.average(loss_dtw)

def save_selected_data(preds_selected, trues_selected, args):
    """
    将选定的数据保存到指定路径中，文件格式为 .npy，文件名称格式化为：
    PeepholeLSTM_<type>_<seq_len>_<pred_len>_<dataset>.npy

    Parameters:
    - preds (np.ndarray): 预测值
    - trues (np.ndarray): 真实值
    - args: 包含 seq_len, pred_len, dataset 等参数
    """
    # 创建保存目录
    save_dir = os.path.join(os.getcwd(), "Draw", "npy")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 文件命名

    pred_file_name = f"{args.model}_pred_{args.seq_len}_{args.pred_len}_{args.dataset}.npy"
    true_file_name = f"{args.model}_true_{args.seq_len}_{args.pred_len}_{args.dataset}.npy"

    # 保存路径
    pred_file_path = os.path.join(save_dir, pred_file_name)
    true_file_path = os.path.join(save_dir, true_file_name)

    # 保存数据
    np.save(pred_file_path, preds_selected)
    np.save(true_file_path, trues_selected)

    print(f"预测数据已保存至: {pred_file_path}")
    print(f"真实数据已保存至: {true_file_path}")


def save_output_to_file(log_file_path):
    # 打开文件并将标准输出重定向到该文件
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

def train(args, model, times, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, NLL_fn, device):
    # 动态生成日志文件路径
    log_file_path = f"Log/training_log_model_name_{args.model}_{args.seq_len}_pred{args.pred_len}_stride{args.stride_len}_alpha{args.alpha}_lr{args.lr}_beta{args.beta}_dataset_{args.dataset}.txt"

    # 配置 logging
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.DEBUG)  # 设置最低级别为 DEBUG，便于捕捉所有日志

    # 清除之前的处理器（如果有）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器，使用仅记录消息的格式
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter('%(message)s')  # 仅记录消息
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)  # 仅记录 INFO 及以上级别的日志

    # 创建控制台处理器，包含时间戳和级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)  # 记录所有级别的日志到控制台

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 使用 logger 记录初始信息
    initial_message = 'Starting training for model:\n\n' + str(model) + '\n\n'
    logger.info(initial_message)
    tqdm_range = tqdm(range(args.epoch))
    tqdm_range.write(initial_message)  # 同时在控制台显示

    # 内存基线
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    num_epochs = args.epoch
    breaking = False

    best_loss = np.inf
    best_mse_plus_dtw = float('inf')  # 初始化值

    for epoch in tqdm_range:
        if breaking:
            break
        model.train()
        start_time = time.time()
        total_dataset_size = 0
        train_mse = []
        train_dt = []
        train_dtw = []
        train_tdi = []
        best_train_mse = np.inf
        loss_dtw = []
        loss_tdi = []

        for batch_idx, batch in enumerate(train_dataloader):
            loss_tdi_f = []
            loss_dtw_f = []
            if breaking:
                break
            batch_coeffs, batch_x, batch_y = batch
            pred_y, dydt = model(args, batch_x, batch_coeffs, times)

            mse = loss_fn(pred_y, batch_y)
            batch_y_epi = batch_y[:, 1:, :]
            batch_y_pre = batch_y[:, :-1, :]
            batch_y_diff = batch_y_epi - batch_y_pre

            loss_dtw, loss_tdi = 0, 0
            batch_size = batch_y.shape[0]
            features = batch_y.shape[-1]
            for f in range(features):
                loss_tdi_ = 0
                loss_dtw_ = 0
                for k in range(batch_size):
                    target_k_cpu = batch_y[k, :, f].view(-1).detach().cpu().numpy()
                    output_k_cpu = pred_y[k, :, f].view(-1).detach().cpu().numpy()

                    # 检查序列的有效性
                    if target_k_cpu.size == 0 or output_k_cpu.size == 0:
                        logger.warning(f"Empty sequence detected at batch {batch_idx}, sample {k}, feature {f}. Skipping DTW.")
                        continue
                    if np.isnan(target_k_cpu).all() or np.isnan(output_k_cpu).all():
                        logger.warning(f"All NaNs in sequence detected at batch {batch_idx}, sample {k}, feature {f}. Skipping DTW.")
                        continue
                    if np.isnan(target_k_cpu).any() or np.isnan(output_k_cpu).any():
                        logger.warning(f"NaNs detected in sequence at batch {batch_idx}, sample {k}, feature {f}. Skipping DTW.")
                        continue

                    try:
                        path, sim = dtw_path(target_k_cpu, output_k_cpu)
                        loss_dtw_ += sim

                        Dist = 0
                        for i, j in path:
                            Dist += (i - j) * (i - j)
                        loss_tdi_ += Dist / (args.pred_len * args.pred_len)
                    except ValueError as e:
                        logger.error(f"DTW computation error at batch {batch_idx}, sample {k}, feature {f}: {e}")
                        continue

                loss_tdi_f.append(loss_tdi_ / batch_size if batch_size > 0 else 0)
                loss_dtw_f.append(loss_dtw_ / batch_size if batch_size > 0 else 0)

            loss_dtw = np.average(loss_dtw_f) if loss_dtw_f else 0
            loss_tdi = np.average(loss_tdi_f) if loss_tdi_f else 0
            dt_loss = loss_fn(dydt, batch_y_diff)

            if np.isnan(mse.item()):
                breaking = True
                logger.error(f"NaN detected in MSE at epoch {epoch}, batch {batch_idx}. Stopping training.")
            loss = (args.alpha * mse) + (args.beta * dt_loss)
            loss.backward()

            # max_norm = 5  # 您可根据需要调整
            utils.clip_grad_norm_(model.parameters(), args.norm)

            optimizer.step()
            optimizer.zero_grad()

            b_size = batch_y.size(0)

            total_dataset_size += b_size
            train_mse.append(mse.item())
            train_dt.append(dt_loss.item())
            train_dtw.append(loss_dtw)
            train_tdi.append(loss_tdi)

        # 计算平均损失
        train_mse = np.average(train_mse) if train_mse else 0
        train_dtw = np.average(train_dtw) if train_dtw else 0
        train_dt = np.average(train_dt) if train_dt else 0
        train_tdi = np.average(train_tdi) if train_tdi else 0

        # 更新最佳训练 MSE
        if train_mse * 1.0001 < best_train_mse:
            best_train_mse = train_mse

        # 构造并记录训练信息
        train_message = (
            f'Epoch: {epoch}  Train MSE: {train_mse:.4f}, '
            f'Train DTW : {train_dtw:.4f}, Train dT : {train_dt:.4f} '
            f'Train TDI: {train_tdi:.4f} Time :{time.time() - start_time:.4f}'
        )
        logger.info(train_message)  # 记录到文件

        # 计算内存使用
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory if baseline_memory else "N/A"

        # 验证和测试
        val_mse, val_dtw, val_dt, val_tdi, val_loss = evaluate(epoch, model, optimizer, val_dataloader, args.model,
                                                               times, loss_fn, NLL_fn)
        test_mse, test_dtw, test_dt, test_tdi, test_loss = evaluate(epoch, model, optimizer, test_dataloader,
                                                                    args.model, times, loss_fn, NLL_fn)

        # 检查是否保存最佳模型
        save_model = False
        if test_mse * 30 + test_dtw * 2.5 + test_tdi < best_mse_plus_dtw:  # 如果当前的test_mse + test_dtw更小
            best_mse_plus_dtw = test_mse * 30 + test_dtw * 2.5 + test_tdi
            save_model = True

        if save_model:
            # 构造保存路径
            model_save_path = (
                f"trained_model/LSTMTIME/{args.dataset}/"
                f"{args.seq_len}_{args.model}_{args.pred_len}_{args.stride_len}_{args.note}_{args.lr}_{args.alpha}_{args.beta}"
            )

            # 确保保存路径存在
            os.makedirs(model_save_path, exist_ok=True)

            # 删除旧的模型文件（如果存在）
            for old_model_file in os.listdir(model_save_path):
                old_model_path = os.path.join(model_save_path, old_model_file)
                if old_model_file.endswith('.pth'):
                    delete_message = f"Deleting old model: {old_model_path}"
                    logger.debug(delete_message)  # 记录为 DEBUG 级别，不会出现在文件中
                    os.remove(old_model_path)

            # 保存当前的模型
            model_filename = f"model_epoch_{epoch}_mse_{test_mse:.4f}_dtw_{test_dtw:.4f}_tdi_{test_tdi:.4f}.pth"
            model_path = os.path.join(model_save_path, model_filename)

            # 保存模型的状态字典和优化器的状态字典
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mse': test_mse,
                'best_dtw': test_dtw
            }, model_path)
            save_message = f"Model saved to {model_path}"
            logger.debug(save_message)  # 记录为 DEBUG 级别，不会出现在文件中

        # 构造并记录验证和测试信息
        val_message = (
            f'Epoch: {epoch}   Validation MSE: {val_mse:.4f}, Validation DTW : {val_dtw:.4f}, '
            f'Validation dT : {val_dt:.4f} TDI: {val_tdi:.4f} Time :{time.time() - start_time:.4f}'
        )
        test_message = (
            f'Epoch: {epoch}   Test MSE: {test_mse:.4f}, Test DTW : {test_dtw:.4f}, '
            f'Test dT: {test_dt:.4f} TestTDI: {test_tdi:.4f} Time :{time.time() - start_time:.4f}'
        )
        memory_message = f"memory_usage:{memory_usage}"

        logger.info(val_message)
        logger.info(test_message)
        logger.info(memory_message)


def main(model_name=args.model, num_epochs=args.epoch):
    manual_seed = args.seed
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    print(f"Setting of this Experiments {args}")
    # args.training = False
    device = "cuda"

    train_dataloader, val_dataloader, test_dataloader, input_channels, output_channels = CPLSTM_dataset.get_dataset(
        args, device)
    model = control_tower.Model_selection_part(args, input_channels=input_channels, output_channels=output_channels,
                                               device=device)
    times = torch.Tensor(np.arange(args.seq_len))
    if args.pretrained:
        load_model(args)
        exit()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    NLL_fn = torch.nn.NLLLoss()
    if args.training:

        ckpt_file = train(args, model, times, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn,
                          NLL_fn, device)
    else:
        MODEL_PATH = '../LSTMCONTIME/trained_model/LSTMTIME' + '/' + str(args.dataset) + '/' + str(
            args.seq_len) + "_" + str(args.model) + "_" + str(args.pred_len) + "_" + str(args.stride_len) + "_" + str(args.note) + "_" + str(
            args.lr) + "_" + str(args.alpha) + "_" + str(args.beta) + "/"
        files_in_directory = os.listdir(MODEL_PATH)

        # 假设我们要查找以"model_epoch"开头的文件
        for file_name in files_in_directory:
            if file_name.startswith('model'):
                # 找到文件并返回完整路径
                full_file_path = os.path.join(MODEL_PATH, file_name)
                print(f"找到文件: {full_file_path}")
                break
        else:
            print("没有找到符合条件的文件")
        print("============> Evaluation <============")
        mae, mse, rmse, mape, mspe, rse, corr, tdi, dtw = load_model(args, full_file_path,
                                                                     visualize_version=args.visualize_version)

        print(
            "Final Results MAE: {:.4f} MSE: {:.4F} RMSE: {:.4f} MAPE: {:.4f} MSPE: {:.4f} RSE: {:.4f} TDI: {:.4f} DTW: {:.4f}".format(
                mae, mse, rmse, mape, mspe, rse, tdi, dtw))


if __name__ == '__main__':
    main()
