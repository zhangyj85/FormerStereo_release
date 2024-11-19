import os
import logging

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING


class ColorLogger():
    def __init__(self, log_dir='.', log_name='temp.log'):
        # set log
        self._logger = logging.getLogger(log_name)              # 创建 log 文件
        self._logger.setLevel(
            logging.INFO if os.environ['LOCAL_RANK']=='0' else logging.WARN
        )                                                       # 记录信息等级(debug信息被忽略)
        log_file = os.path.join(log_dir, log_name)              # log 文件路径
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(                          # 输出格式: 时间 信息
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")

        file_log = logging.FileHandler(log_file, mode='a')      # 创建文件句柄, 追加方式写入
        file_log.setLevel(logging.INFO)                         # 文件写入内容的等级
        file_log.setFormatter(formatter)

        console_log = logging.StreamHandler()                   # 创建屏幕句柄
        console_log.setLevel(logging.INFO)                      # 屏幕输出内容的等级
        console_log.setFormatter(formatter)

        # 将文件句柄和屏幕句柄添加到 logger 中
        self._logger.addHandler(file_log)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)

    def _print_configures(self, cfg):
        print_configures(self, cfg)

def yellow(msg):
    return YELLOW + str(msg) + END

def green(msg):
    return GREEN + str(msg) + END

def red(msg):
    return RED + str(msg) + END

def blue(msg):
    return BLUE + str(msg) + END

def print_yellow(msg, **kwargs):
    print(yellow(msg), **kwargs)

def print_green(msg, **kwargs):
    print(green(msg), **kwargs)

def print_red(msg, **kwargs):
    print(red(msg), **kwargs)

def print_blue(msg, **kwargs):
    print(blue(msg), **kwargs)

def error(msg):
    print(red('ERR: ' + str(msg)))

def warning(msg):
    print(yellow('WRN: ' + str(msg)))


def print_configures(logger, config):
    # environment setting
    logger.info("{} Model Configures {}".format("*"*20, "*"*20))   # 输出表头
    logger.info("Mode: {}\t Split: {}\t".format(config['mode'], config['split']))

    # dataset
    logger.info("Dataset: {}".format(config['data']['datasets']))
    logger.info("Image Mean: {}".format(config['data']['mean']))
    logger.info("Image Std: {}".format(config['data']['std']))

    # model
    logger.info("Model Name: {}".format(config['model']['name']))
    logger.info("Proposal Disparity: {} -> {}".format(config['model']['min_disp'], config['model']['max_disp']))

    # solver
    if config['mode'].lower() == "train":
        logger.info("Augment: {}".format(config['data']['augment']))
        logger.info("Crop[H, W]: {}".format(config['data']['crop_size']))
        logger.info("Batch Size: {}".format(config['train']['batch_size']))
        logger.info("Total Epochs: {}".format(config['train']['epoch']))
        logger.info("Weights Save In: {}".format(os.path.join(config['train']['save_path'], config['model']['name'])))
