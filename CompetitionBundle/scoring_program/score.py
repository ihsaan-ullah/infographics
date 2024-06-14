# ------------------------------------------
# Imports
# ------------------------------------------
import os
import io
import re
import sys
import json
import base64
import requests
import numpy as np
from PIL import Image
from datetime import datetime as dt
from constants import GPT_KEY


# ------------------------------------------
# Settings
# ------------------------------------------
CODABENCH = True  # True when running on Codabench


class Scoring:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.scores_dict = {}

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
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def set_directories(self):

        # set default directories for Codabench
        module_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir_name = os.path.dirname(module_dir)

        score_file_name = "scores.json"
        html_file_name = "detailed_results.html"
        html_template_file_name = "template.html"

        output_dir_name = "scoring_output"
        reference_dir_name = "reference_data"
        predictions_dir_name = "sample_result_submission"

        if CODABENCH:
            root_dir_name = "/app"
            output_dir_name = "output"
            reference_dir_name = 'input/ref'
            predictions_dir_name = 'input/res'

        # Directory to output computed score into
        self.output_dir = os.path.join(root_dir_name, output_dir_name)
        # reference data (test labels)
        self.reference_dir = os.path.join(root_dir_name, reference_dir_name)
        # submitted/predicted labels
        self.prediction_dir = os.path.join(root_dir_name, predictions_dir_name)

        # score file to write score into
        self.score_file = os.path.join(self.output_dir, score_file_name)
        # html file to write score and figures into
        self.html_file = os.path.join(self.output_dir, html_file_name)
        # html template firle
        self.html_template_file = os.path.join(module_dir, html_template_file_name)

        # Add to path
        sys.path.append(self.reference_dir)
        sys.path.append(self.output_dir)
        sys.path.append(self.prediction_dir)

    def load_image_captions(self):
        print("[*] Loading image captions")
        input_json_file = os.path.join(self.reference_dir, "image_captions.json")
        with open(input_json_file, 'r') as file:
            self.image_captions = json.load(file)

    def load_reference_data(self):
        print("[*] Reading reference data")
        try:
            self.reference_images = os.listdir(self.reference_dir)
        except:
            raise ValueError("[-] Failed to read reference data")

    def load_ingestion_result(self):
        print("[*] Reading generated data")
        try:
            self.generated_images = os.listdir(self.prediction_dir)
        except:
            raise ValueError("[-] Failed to read generated data")

    def _resize_image_to_height(self, image, target_height):
        """
        Resizes an image to a target height while maintaining aspect ratio.
        """
        width, height = image.size
        new_width = int(target_height * width / height)
        return image.resize((new_width, target_height), Image.Resampling.LANCZOS)

    def _get_GPT_Feedback(self, base64_img, image_caption):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GPT_KEY}"
        }

        prompt = f"In the attached image there are two infographics generated for a caption: '{image_caption}'. Compare the two infographics and return a score 1 if the infographic on the right is better than that on the left to illustrate the caption, and 0 otherwise. Give the score at the end in a separate line in this format 'Score: score_value'"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            review = response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(response)
            print(e)

        return review

    def get_feedback_from_LLM(self):
        print("[*] Getting feedback from LLM")
        dict_list = []
        scores = []
        for image_caption in self.image_captions:
            image_name = f"{image_caption.replace(' ', '-')}.png"

            # create one image: left reference, right generated

            space = 50
            max_size_mb = 15

            reference_image_path = os.path.join(self.reference_dir, image_name)
            generated_image_path = os.path.join(self.prediction_dir, image_name)
            reference_image = Image.open(reference_image_path)
            generated_image = Image.open(generated_image_path)

            # Determine the new height (max height of both images)
            new_height = max(reference_image.height, generated_image.height)

            # Resize images to the new height while maintaining aspect ratio
            reference_image = self._resize_image_to_height(reference_image, new_height)
            generated_image = self._resize_image_to_height(generated_image, new_height)

            # Determine the size of the new image
            new_width = reference_image.width + generated_image.width + space

            # Create a new image with a white background
            combined_image = Image.new('RGB', (new_width, new_height), 'white')

            # Paste the two images into the new image
            combined_image.paste(reference_image, (0, 0))
            combined_image.paste(generated_image, (reference_image.width + space, 0))

            output_path = os.path.join(self.output_dir, image_name)

            # Check the size of the combined image and resize if necessary
            buffered = io.BytesIO()
            combined_image.save(buffered, format="PNG")
            img_size = buffered.tell()

            # Reduce dimensions if the image size is greater than 15 MB
            while img_size > max_size_mb * 1024 * 1024:
                new_width = int(new_width * 0.9)
                new_height = int(new_height * 0.9)
                combined_image = combined_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save to buffer again to check size
                buffered = io.BytesIO()
                combined_image.save(buffered, format="PNG")
                img_size = buffered.tell()

            # remove this for production
            combined_image.save(output_path)

            # convert image to base 64
            base_64_combined_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Get GPT review and score
            gpt_review = self._get_GPT_Feedback(base_64_combined_img, image_caption)

            # Extract score from review
            try:
                score_pattern1 = r"Score:\s*([0-9]+(?:\.[0-9]+)?)"
                score_pattern2 = r"\*\*Score\*\*:\s*([0-9]+(?:\.[0-9]+)?)"
                match1 = re.search(score_pattern1, gpt_review)
                match2 = re.search(score_pattern2, gpt_review)
                if match1:
                    score = match1.group(1)
                    gpt_review = re.sub(r"Score:.*(\n|$)", "", gpt_review)
                elif match2:
                    score = match2.group(1)
                    gpt_review = re.sub(r"**Score**:.*(\n|$)", "", gpt_review)
            except:
                score = -1
            score = int(score)

            print(f"\tCaption: {image_caption}\n\tScore: {score}\n")
            scores.append(score)
            dict_list.append({
                "caption": image_caption,
                "score": score,
                "review": gpt_review
            })

        self.scores_dict = {
            "avg_score": np.mean(scores),
            "scores_details": dict_list
        }

    def write_scores(self):
        print("[*] Writing scores")
        with open(self.score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))


if __name__ == "__main__":
    print("############################################")
    print("### Starting Result Compilation Program")
    print("############################################\n")

    # Init scoring
    scoring = Scoring()

    # Set directories
    scoring.set_directories()

    # Start timer
    scoring.start_timer()

    # Load image captions
    scoring.load_image_captions()

    # Load reference data
    scoring.load_reference_data()

    # Load ingestion result
    scoring.load_ingestion_result()

    # Getting feedback from LLM
    scoring.get_feedback_from_LLM()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scoring program executed successfully!")
    print("----------------------------------------------\n\n")
