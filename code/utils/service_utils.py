import mysql.connector

from utils.config_utils import ConfigUtils

config_file = "process_control_config.yaml"
config_dict = ConfigUtils.get_config_dict(config_file)


def get_mysql_conn():
    return mysql.connector.connect(
        host=config_dict["host"],
        user=config_dict["user"],
        password=config_dict["pwd"],
        database=config_dict["db"],
        auth_plugin='mysql_native_password'
    )
