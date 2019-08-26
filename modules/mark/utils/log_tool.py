import os
import logging.handlers
import logging
from .path_tool import makesurepath

cmd_path = os.getcwd()
basic_path = os.path.join(cmd_path, "..", "..", "..", "data")
project_path = os.path.join(basic_path, "mark")
model_path = os.path.join(project_path, "models")
data_path = os.path.join(project_path, "data")
makesurepath(data_path)
out_path = os.path.join(project_path, "output")
makesurepath(out_path)
conf_path = os.path.join(project_path, "config")
makesurepath(conf_path)
log_path = os.path.join(project_path, 'logs')
makesurepath(log_path)
log_file = os.path.join(log_path, 'mark.log')

# 创建一个logger
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 写入文件，如果文件超过100个Bytes，仅保留5个文件。
fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=104857600, backupCount=10)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
