import os
import csv
from datetime import datetime
from reporters.osc_reporter import OSC_Reporter

class Log_Reporter(OSC_Reporter):
    def __init__(self, ip, send_port):
        super().__init__(ip, send_port)

        # Create logs folder if it does not exist
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Create new CSV file with today's session time
        self.log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

        # Track known column headers
        self.headers = ["timestamp"]

        # Check if the file already exists, otherwise create it
        if os.path.exists(self.log_file):
            with open(self.log_file, mode="r", newline="") as file:
                reader = csv.reader(file)
                # update headers if neccesary
                existing_headers = next(reader, None)
                if existing_headers:
                    self.headers = existing_headers
        else:
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)
        
        # save last timestamp, only update on new stamp
        self.last_time = ""

    def send(self, data_dict):
        send_pairs = self.flatten(data_dict)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if timestamp == self.last_time:
            return send_pairs
        else:
            # update last time stamp
            self.last_time = timestamp

            # Convert to dictionary format
            row_data = {"timestamp": timestamp}
            for key, value in send_pairs:
                row_data[key] = value

            # Ensure headers include all keys, including new ones
            new_keys = set(row_data.keys()) - set(self.headers)
            if new_keys:
                self.headers.extend(new_keys)
                self._update_csv_headers()

            # Write the row
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                writer.writerow(row_data)

        return send_pairs

    def _update_csv_headers(self):
        temp_file = self.log_file + ".tmp"

        with open(self.log_file, mode="r", newline="") as infile, open(temp_file, mode="w", newline="") as outfile:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=self.headers)
            
            # Write new header
            writer.writeheader()
            
            # Rewrite existing rows with new headers (filling missing values)
            for row in reader:
                writer.writerow({key: row.get(key, "") for key in self.headers})

        os.replace(temp_file, self.log_file)
