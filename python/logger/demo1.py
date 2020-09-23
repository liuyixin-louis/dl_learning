import logging
import sys 

def get_logger():
    logger = logging.getLogger('logger_name')
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s') # 时间戳、等级、消息
    
    console_formatter = logging.Formatter('%(message)s')


    file_handler = logging.FileHandler('basic\logger\log\\a.log')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    logger = get_logger()
    logger.warning("this is a logger info!")
    logger.debug("this is a logger info!")
    logger.info("this is a logger info!")
    

