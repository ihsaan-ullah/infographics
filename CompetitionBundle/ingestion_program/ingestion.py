# ------------------------------------------
# Imports
# ------------------------------------------
import os
import sys
import json
import requests
import warnings
from datetime import datetime as dt
warnings.filterwarnings("ignore")

# ------------------------------------------
# Settings
# ------------------------------------------
CODABENCH = False  # True when running on Codabench
VERBOSE = False  # False for codabench, True for debugging


# ------------------------------------------
# Ingestion Class
# ------------------------------------------
class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        print("\n---------------------------------")
        print(f'[✔] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def set_directories(self):

        # set default directories
        module_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir_name = os.path.dirname(module_dir)

        input_data_dir_name = "input_data"
        output_dir_name = "sample_result_submission"
        program_dir_name = "ingestion_program"
        submission_dir_name = "sample_code_submission"

        if CODABENCH:
            root_dir_name = "/app"
            input_data_dir_name = "input_data"
            output_dir_name = "output"
            program_dir_name = "program"
            submission_dir_name = "ingested_program"

        # Input data directory to read training and test data from
        self.input_dir = os.path.join(root_dir_name, input_data_dir_name)
        # Output data directory to write predictions to
        self.output_dir = os.path.join(root_dir_name, output_dir_name)
        # Program directory
        self.program_dir = os.path.join(root_dir_name, program_dir_name)
        # Directory to read submitted submissions from
        self.submission_dir = os.path.join(root_dir_name, submission_dir_name)

        # Add to path
        sys.path.append(self.input_dir)
        sys.path.append(self.output_dir)
        sys.path.append(self.program_dir)
        sys.path.append(self.submission_dir)

    def load_input_data(self):
        print("[*] Loading image captions")
        input_json_file = os.path.join(self.input_dir, "image_captions.json")
        with open(input_json_file, 'r') as file:
            self.image_captions = json.load(file)

    def init_submission(self):
        print("[*] Initializing submitted model")
        from model import Model
        self.model = Model(image_captions=self.image_captions)

    def generate_images(self):
        print("[*] Generating images from image captions")
        self.images_urls = self.model.generate_images()
        print(self.images_urls)

    def save_images(self):
        print("[*] Saving generated images")
        for image_url, image_caption in zip(self.images_urls, self.image_captions):
            image_name = f"{image_caption.replace(' ', '-')}.png"
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image_path = os.path.join(self.output_dir, image_name)
                with open(image_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)


if __name__ == '__main__':

    print("############################################")
    print("### Starting Ingestion Program")
    print("############################################\n")

    # Init Ingestion
    ingestion = Ingestion()

    ingestion.set_directories()

    # Start timer
    ingestion.start_timer()

    # load input data
    ingestion.load_input_data()

    # init submitted model
    ingestion.init_submission()

    # generate images
    ingestion.generate_images()

    # save images
    ingestion.save_images()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestion program executed successfully!")
    print("----------------------------------------------\n\n")
