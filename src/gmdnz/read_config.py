import configparser

class Config:
    def __init__(self, config_file='config.ini'):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_file)


    def list_variables(self, section_name):
        # Get all keys (option names) from a section
        if self.parser.has_section(section_name):
            keys = self.parser.options(section_name)  # Returns a list of keys
            return keys
        else:
            print(f"Section '{section_name}' not found.")

    def get(self, section, option, fallback=None, dtype=str):
        value = self.parser.get(section, option, fallback=fallback)
        if dtype == list:
            return [item.strip() for item in value.split(",")]
        return dtype(value)

    def getint(self, section, option, fallback=None):
        return self.parser.getint(section, option, fallback=fallback)

    def getfloat(self, section, option, fallback=None):
        return self.parser.getfloat(section, option, fallback=fallback)

    def getboolean(self, section, option, fallback=False):
        return self.parser.getboolean(section, option, fallback=fallback)

# Create a single shared config instance
#config = Config()

