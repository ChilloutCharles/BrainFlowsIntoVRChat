from reporters.osc_reporter import OSC_Reporter

class Debug_Reporter(OSC_Reporter):
    def __init__(self, ip, send_port):
        super().__init__(ip, send_port)
    
    def send(self, data_dict):
        # flatten dictionary into a list of pairs
        send_pairs = self.flatten(data_dict)

        # send each pair
        for path, value in send_pairs:
            self.osc_client.send_message(path, value)
        
        return send_pairs