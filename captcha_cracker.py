import glob
import os

import CaptchaCracker as cc


class CaptchaModel:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(self.current_dir, "models", "weights.h5")

        self.img_width = 130
        self.img_height = 35

        self._learn()

        img_length = 6

        img_char = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

        self.apply_model = cc.ApplyModel(
            self.weights_path,
            self.img_width,
            self.img_height,
            img_length,
            img_char,
        )

    def _learn(self):
        img_path_list = glob.glob(os.path.join(self.current_dir, "samples", "*.png"))

        model = cc.CreateModel(
            img_path_list,
            self.img_width,
            self.img_height,
        ).train_model(epochs=128)

        model.save_weights(self.weights_path)

    def predict_from_bytes(self, buffer):
        prediction = self.apply_model.predict_from_bytes(buffer)

        return prediction
