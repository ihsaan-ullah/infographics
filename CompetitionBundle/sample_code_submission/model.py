from openai import OpenAI
from constants import GPT_KEY


class Model:

    def __init__(self, image_captions):
        self.image_captions = image_captions

    def generate_images(self):
        """
        Generates images using dall e

        Args:
            None

        Returns:
            list: a list of image urls (one for each image caption)
        """

        client = OpenAI(
            api_key=GPT_KEY,
        )
        image_urls = []
        for image_caption in self.image_captions:
            print(f"\t {image_caption}")
            prompt = f"generate an infographic for {image_caption}"
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                quality="standard",
                n=1,
            )

            image_urls.append(
                response.data[0].url
            )

        return image_urls
