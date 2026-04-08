import logging

APP_LOG_LEVEL = logging.INFO + 1
logging.addLevelName(APP_LOG_LEVEL, 'APP_INFO')
def app_info(self, msg, *args, **kwargs):
     if self.isEnabledFor(APP_LOG_LEVEL):
        self._log(APP_LOG_LEVEL, msg, args, **kwargs)
logging.Logger.app_info = app_info
