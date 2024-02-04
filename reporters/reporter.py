from reporters.base_reporter import Base_Reporter

class Reporter(Base_Reporter):
    def __init__(self, reporter_dict={}):
        self.reporters = reporter_dict
    
    def register_reporters(self, reporter_dict):
        self.reporters |= reporter_dict
    
    def register_reporter(self, reporter_name, reporter_obj):
        self.reporters[reporter_name] = reporter_obj
    
    def unregister_reporter(self, reporter_name):
        del self.reporters[reporter_name]

    def send(self, data_dict):
        message_pairs_per_reporter = map(lambda rp: rp.send(data_dict), self.reporters.values())
        message_pairs = sum(message_pairs_per_reporter, [])
        return message_pairs