import telebot
import io
import PIL.Image as Image
import numpy as np
from ..model import _model, _download
import cv2
from pathlib import Path
import warnings


class DetectionBot:
    """
    Class decorator for telebot.TeleBot class. Provides methods for checking and applying functions for image-translator bot.
    """
    def __init__(self, token: str):
        if not isinstance(token, str):
            raise TypeError("Provided token is not valid")
        DetectionBot.check_downloaded()
        self.model_ = _model.FullModel()
        self.bot_ = telebot.TeleBot(token=token)
        DetectionBot._init(self.bot_, self.model_)

    def polling(self, none_stop: bool = True) -> None:
        """
        Function that uses as access to telebot.TeleBot polling method
        :param none_stop:
        :return:
        """
        self.bot_.polling(none_stop=none_stop)

    @staticmethod
    def check_downloaded() -> None:
        """
        Checks whether all the required files are downloaded. If not - calls for download() function from _download.py
        :return:
        """
        paths = ['./detectionbot/model/deploy.prototxt',
                 './detectionbot/model/res10_300x300_ssd_iter_140000.caffemodel',
                 './detectionbot/model/age_gender_models/deploy_age.prototxt',
                 './detectionbot/model/age_gender_models/age_net.caffemodel',
                 './detectionbot/model/age_gender_models/deploy_gender.prototxt',
                 './detectionbot/model/age_gender_models/gender_net.caffemodel']
        if all([Path(p).exists() for p in paths]):
            print('All required files are downloaded. Launching bot...')
        else:
            warnings.warn("Not all required files are downloaded")
            _download.download()

    @staticmethod
    def _init(bot_: telebot.TeleBot, model_: _model.FullModel = None) -> None:
        """
        Initializer for bot`s decorators
        """
        @bot_.message_handler(commands=['start'])
        def welcome(message: telebot.types.Message) -> None:
            """
            Start function /start
            """
            bot_.send_chat_action(message.chat.id, 'typing')
            sti = open('./files/AnimatedSticker.tgs', 'rb')
            bot_.send_sticker(message.chat.id, sti)
            bot_.send_message(message.chat.id,
                              "Добро пожаловать, {0.first_name}!\nЯ - <b>{1.first_name}</b>, бот созданный чтобы быть подопытным кроликом."
                              .format(message.from_user, bot_.get_me()),
                              parse_mode='html')

        @bot_.message_handler(content_types=['text'])
        def rep(message: telebot.types.Message) -> None:
            """
            Just echo
            """

            bot_.send_message(message.chat.id, message.text)

        @bot_.message_handler(content_types=['photo'])
        def photo(message: telebot.types.Message) -> None:
            """
            Saves sended photo to the log directory, applies model to sended photo and sends reconstructed photo back.
            """
            if not model_:
                raise ValueError("No model for prediction provided")
            bot_.send_message(message.chat.id, 'фото получено')
            file_id = message.photo[-1].file_id
            file_info = bot_.get_file(file_id)
            download_file = bot_.download_file(file_info.file_path)
            image = np.array(Image.open(io.BytesIO(download_file)))
            boxes, genders, ages = model_.predict(image)
            img_c = _model.transform(image, boxes, genders, ages)
            img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
            path = f'./images/{message.chat.id}.jpg'

            cv2.imwrite(path, img_c)

            with open(path, 'rb') as f:
                bot_.send_photo(message.chat.id, f)

if __name__ == '__main__':
    pass
