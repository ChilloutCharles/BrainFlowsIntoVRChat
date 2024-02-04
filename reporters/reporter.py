from reporters.base_reporter import Base_Reporter

class Reporter(Base_Reporter):
    def __init__(self, reporters):
        self.reporters = reporters
    
    def send(self, data_dict):
        message_pairs_per_reporter = map(lambda rp: rp.send(data_dict), self.reporters)
        message_pairs = sum(message_pairs_per_reporter, [])
        return message_pairs


        
    