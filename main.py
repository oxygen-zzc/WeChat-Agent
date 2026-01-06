# 完整的微信自动化聊天，从打开微信查找联系人开始
import json
import time

import cv2
import numpy as np
import pyautogui
import pyperclip

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from detection_tools import open_wechat
from detection_tools import click_search
from detection_tools import get_retina_scale_factor
from detection_tools import screen_capture_compress

vl_model, vl_processor = load("mlx-community/Qwen3-VL-8B-Instruct-8bit")
vl_config = load_config("mlx-community/Qwen3-VL-8B-Instruct-8bit")

# 点击聊天框
def click_chat_field():
    print(f"📅 VL识别聊天输入框开始：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    scale_factor = get_retina_scale_factor()
    # 先截图，再压缩，直接按屏幕比例进行压缩
    screenshot = screen_capture_compress(1/scale_factor)
    image = [screenshot]
    prompt = "给出微信聊天输入框窗口的坐标，返回JSON，参数：x1y1(左上角),x2y2(右下角)，只返回JSON，不要其他文字内容，如果解析失败，返回空JSON"

    # Apply chat template
    formatted_prompt = apply_chat_template(
        vl_processor, vl_config, prompt, num_images=1
    )

    output = generate(vl_model, vl_processor, formatted_prompt, image)
    result = output.text
    print(f"聊天输入框位置：{result}")
    print(f"📅 VL识别聊天输入框结束：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    result_json = json.loads(result)
    x1y1 = result_json["x1y1"]
    x2y2 = result_json["x2y2"]
    pos = ((x1y1[0]+x2y2[0])/2, (x1y1[1]+x2y2[1])/2)
    print(pos)
    pyautogui.click(pos)

# 识别对方的消息
# def read_message():


def main():
    # chat_name = input("联系人:").strip()
    # chat_message = input("消息内容:").strip()
    chat_name = ""
    chat_message = ""

    # 启动微信
    open_wechat()
    # 等待0.2s
    time.sleep(0.2)
    # OCR识别搜索框，并打开
    click_search()
    # 输入联系人，pyautogui直接输入中文有问题，这里用剪贴板的方式
    pyperclip.copy(chat_name)
    pyautogui.press('esc')
    pyautogui.hotkey("command", "v")
    pyautogui.press('enter')
    click_chat_field()
    # 粘贴输入的文字内容
    pyperclip.copy(chat_message)
    pyautogui.hotkey("command", "v")
    pyautogui.press('enter')

    while True:
        time.sleep(30)


if __name__ == '__main__':
    main()