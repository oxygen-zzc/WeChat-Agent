# 单一的微信聊天工具，截取当前屏幕最后一条对方消息进行聊天

import pyautogui
import pyperclip
import time

from detection_tools import screen_capture_compress
from vlm_util import VLMProcessor
from lm_util import LMProcessor


def main():
    # 截图，获取最后一条对方的消息
    # time.sleep(1)
    # screen = screen_capture_compress()
    # read_message(screen)

    # 模型初始化
    lm = LMProcessor()
    vlm = VLMProcessor()

    last_msg = None
    while True:
        time.sleep(30)
        screen = screen_capture_compress()
        msg = vlm.read_message(screen)
        if msg is None:
            print("未识别到消息")
            continue
        if msg == last_msg:
            print("未识别到新消息")
            continue

        last_msg = msg
        # LLM回复消息
        res = lm.chat(msg)
        if res is None:
            print("error")
            continue

        # 消息写入微信
        pyperclip.copy(res)
        pyautogui.press("esc")
        pyautogui.hotkey("command", "v")
        pyautogui.press("enter")


if __name__ == '__main__':
    main()