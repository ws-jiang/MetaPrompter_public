import socket
from enum import Enum


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class ExptEnum(Enum):
    S_5_WAY_1_SHOT = "5way1shot"
    H_5_WAY_1_SHOT = "h5way1shot"
    S_5_WAY_5_SHOT = "5way5shot"
    H_5_WAY_5_SHOT = "h5way5shot"
    S_10_SHOT = "10shot"
    S_30_SHOT = "30shot"
    S_15_SHOT = "15shot"
    S_5_SHOT = "5shot"
    S_2_SHOT = "2shot"

class ProgressConfig(object):
    def __init__(self, config):
        self.debug_log_freq = config.get("debug_log_freq", 10)
        self.max_not_impr_cnt = config.get("max_not_impr_cnt", 5)
        self.info_log_freq = config.get("info_log_freq", 100)
        self.eval_freq = config.get("eval_freq", 1000)
        self.meta_valid_freq = config.get("meta_valid_freq", 1000)
        self.report_train_2_mysql_freq = config.get("report_train_2_mysql_freq", 1000)
        self.num_workers = config.get("num_workers", 8)