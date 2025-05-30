from PIL import Image
from abc import ABC, abstractmethod
import numpy as np
import pytesseract

class OCR_Runner(ABC):
    def __init__(self, lang='eng+kor'):
        """
        lang: Tesseract language code, e.g., 'eng', 'kor', 'eng+kor'
        """
        self.lang = lang

    def run_on_image(self, np_image: np.ndarray) -> list[dict]:
        """
        Run OCR on a cropped image region.

        Args:
            np_image (np.array): Cropped numpy image, shape (H, W, 3), uint8

        Returns:
            List[dict]: [{'text': str, 'bbox': [x0, y0, x1, y1]}]
        """
        if np_image.shape[2] != 3 or np_image.dtype != np.uint8:
            raise ValueError("Input must be uint8 RGB/BGR image array")

        pil_image = Image.fromarray(np_image)

        data = pytesseract.image_to_data(pil_image, lang=self.lang, output_type=pytesseract.Output.DICT)

        results = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text == '':
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            bbox = [x, y, x + w, y + h]
            results.append({'text': text, 'bbox': bbox})

        return results


if __name__ == "__main__":
    ocr_runner = OCR_Runner()
    np_image = np.array(Image.open("/media/works/pdf2pdf/terra-pdf/test_pdf/english_png_sample.png").convert("RGB"))
    results = ocr_runner.run_on_image(np_image)
    # print(results)
    for result in results:
        # print(result)
        print(result['text'])
        # print(result['bbox'])
        print("-"*100)