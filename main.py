import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from adapt import get_method
from utils import datasets
from utils.misc import set_global_seeds, save_configuration


def argparser():
    parser = argparse.ArgumentParser(
        "Weight Average Test Time Adaptation of CLIP")

    # Directories
    parser.add_argument('--data_dir',
                        type=str,
                        default='/export/livia/home/vision/Mnoori/data/',
                        help='Root directory for datasets')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='save/',
        help='Path for saving base training weights and results')

    # General settings
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility')

    # Model
    parser.add_argument('--backbone',
                        type=str,
                        default='ViT-B/32',
                        help='Model backbone to use')

    # Dataset settings
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=('cifar10', 'cifar100', 'tiny-imagenet',
                                 'visda', 'PACS', 'office_home', 'VLCS'),
                        help='Dataset to use')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Number of workers for data loading')

    # Training settings
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        help='Batch size for training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate')
    parser.add_argument('--trials',
                        default=3,
                        type=int,
                        help='Number of trials to repeat the experiments')

    # Evaluation settings
    parser.add_argument('--adapt',
                        action='store_true',
                        help='Enable adaptation')

    # Corruptions settings
    parser.add_argument(
        '--corruptions_list',
        nargs='+',
        default=None,
        type=str,
        help='List of corruptions to apply to the dataset (Cifar datasets)')

    # Method name
    parser.add_argument('--method',
                        type=str,
                        default='watt',
                        choices=('watt'),
                        help='Method to use for adaptation')

    return parser


def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'watt':
        parser.add_argument(
            '--watt_type',
            type=str,
            default='sequential',
            choices=('parallel', 'sequential'),
            help='Type of WATT adaptation (parallel or sequential)')
        parser.add_argument(
            '--watt_l',
            default=2,
            type=int,
            help=
            'Number of adaptation iterations for each text embedding before weight averaging'
        )
        parser.add_argument(
            '--watt_m',
            default=5,
            type=int,
            help=
            'Number of repetitions of the adaptation and weight averaging process'
        )
        parser.add_argument('--watt_temps',
                            type=str,
                            default='templates.yaml',
                            help='Path to the templates.yaml file')
        parser.add_argument(
            '--watt_reference_for_evaluation',
            action='store_true',
            help=
            'Use REFERENCE_TEMPLATE during evaluation instead of averaging text embeddings of different templates'
        )

    # Add other methods here
    else:
        raise ValueError(f"Unknown method: {method}")

    return parser


def main():
    # Initial argument parsing to get the method
    # 调用 argparser() 从命令行获取参数
    initial_parser = argparser()
    # initial_args 获取已知参数，未知参数不关心
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    # 再次调用 argparser() 构造一个新的基础 parser
    parser = argparser()
    # 使用 add_method_specific_args(parser, method) 函数向 parser 中添加与指定 method 相关的参数，
    # 例如某个方法可能需要 --lr, --batch-size, --momentum 等。
    parser = add_method_specific_args(parser, initial_args.method)
    # 最终调用 parse_args() 解析所有参数并存入 args
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    # 设置随机种子，确保实验结果可复现
    set_global_seeds(args.seed)

    # Save the configuration settings
    # 保存当前配置参数（可能用于记录日志、实验对比、复现实验等）
    save_configuration(args)

    # 根据当前设备是否支持 CUDA 来确定使用 GPU 还是 CPU 进行计算
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setting up the model and the method
    # 根据 args.method 构造相应的自适应方法实例（如 TENT、SHOT、CoTTA 等）
    # get_method(args, device) 是一个工厂函数，根据方法名（例如 args.method == "TENT"）返回对应的方法类实例 adapt_method，并将其部署到 device 上
    adapt_method = get_method(args, device)

    # 指定一个结果输出路径，例如将最终精度或日志保存为 results.txt 文件
    results_path = os.path.join(args.save_dir, "results.txt")

    # args.corruptions_list 是一个腐扰名称列表，如 ['gaussian_noise', 'motion_blur', 'brightness']
    # 对每个腐扰类型进行完整的一轮评估试验
    for corruption in args.corruptions_list:
        # 调用 datasets.prepare_data(...) 加载指定腐扰类型的数据
        # data_loader: 包含腐扰样本的 DataLoader
        # classes: 当前任务的类别信息（如 imagenet 的 1000 类标签）
        data_loader, classes = datasets.prepare_data(
            args.dataset,
            args.data_dir,
            corruption=corruption,
            batch_size=args.batch_size,
            num_workers=args.workers)
        # 存放每次 trial 的准确率
        acc = []
        # 每种 corruption 类型下重复测试 args.trials 次（如 3 次），减少偶然性影响
        for t in range(args.trials):
            # 初始化准确率统计
            correct = 0
            # 从 DataLoader 中取数据，遍历数据集的每一个 batch
            # inputs, labels: 一个 batch 的图像和标签
            for batch_idx, (inputs, labels) in tqdm(enumerate(data_loader),
                                                    total=len(data_loader)):
                # 将数据转到 GPU（如果可用）
                inputs, labels = inputs.to(
                    device, non_blocking=True), labels.to(device,
                                                          non_blocking=True)

                # reset the model before adapting to a new batch
                # 为了公平起见，在每个 batch 上重新初始化适应状态
                adapt_method.reset()

                # perform adaptation
                # 如果设置了 --adapt 参数，就调用 adapt_method.adapt() 执行 TTA
                # 这一步会更新模型参数
                if args.adapt:
                    adapt_method.adapt(inputs, classes)

                # perform evaluation
                # evaluate 是模型的前向推理函数
                # 此行使模型进行推理
                pred = adapt_method.evaluate(inputs, classes)

                # Calculate the number of correct predictions
                # 若 pred 是 [1, B]，且 labels 是 [B]，就先 view 成 [1, B] 再 expand 以对齐维度。
                # eq() 得到布尔矩阵，表示每个预测是否正确
                correctness = pred.eq(labels.view(1, -1).expand_as(pred))
                # .sum().item()：统计所有预测正确的个数并转换成数值
                # correct：当前试验累计的正确预测数量
                correct += correctness.sum().item()
                print(correct)

            # 输出每个试验的准确率
            acc.append(correct / len(data_loader.dataset))
            print(correct / len(data_loader.dataset))

        # 输出多次试验的平均准确率与标准差
        print(
            str(round(np.array(acc).mean() * 100, 2)) + ',' +
            str(round(np.array(acc).std() * 100, 2)))
        # 保存结果到文件
        with open(results_path, 'w') as fichier:
            fichier.write(
                str(round(np.array(acc).mean() * 100, 2)) + ',' +
                str(round(np.array(acc).std() * 100, 2)) + '\n')


if __name__ == "__main__":
    main()
