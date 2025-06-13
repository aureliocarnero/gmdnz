import configparser

class Config:
    def __init__(self, config_file='config.ini'):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_file)

    def get(self, section, option, fallback=None):
        return self.parser.get(section, option, fallback=fallback)

    def getint(self, section, option, fallback=None):
        return self.parser.getint(section, option, fallback=fallback)

    def getfloat(self, section, option, fallback=None):
        return self.parser.getfloat(section, option, fallback=fallback)

    def getboolean(self, section, option, fallback=None):
        return self.parser.getboolean(section, option, fallback=fallback)

# Create a single shared config instance
#config = Config()

