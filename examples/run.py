import sys
sys.path.append('..')
from detectionbot.bot._config import Parser
from detectionbot.bot._bot import DetectionBot

if __name__ == '__main__':
    # simple usage: python run.py <TOKEN>
    TOKEN = Parser().parse_args
    MyBot = DetectionBot(token=TOKEN)
    MyBot.polling(none_stop=True)




