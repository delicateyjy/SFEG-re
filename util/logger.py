"""
存储日志
"""
import logging

def get_logger(process_log_path, name):
    logger = logging.getLogger(name)
    filename = f'{process_log_path}/{name}.log'

    fh = logging.FileHandler(filename, mode='a', encoding='utf-8') #w+：读写模式，a：续写
    # 格式：时间戳 Logger名称 日志级别 日志消息
    # 示例：2025-09-28 22:12:39,575 eval INFO mIouU -> 0.5053484368701702
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    # 设置日志级别为DEBUG
    # 级别优先级：DEBUG < INFO < WARNING < ERROR < CRITICAL
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger