import json
import time
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

class VLMProcessor:
    def __init__(self):
        self.vl_model, self.vl_processor = load("mlx-community/Qwen3-VL-8B-Instruct-8bit")
        self.vl_config = load_config("mlx-community/Qwen3-VL-8B-Instruct-8bit")

    def read_message(self,frame):
        print(f"📅 VL识别对方消息开始：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        image = [frame]
        prompt = "给出微信聊天框中对方发送的最新一条消息，注意在界面中越靠下越新，靠左侧的白色聊天框是对方发送的消息，返回JSON，参数：msg(消息内容)，只返回JSON，不要其他文字内容，如果解析失败或者对方没有发送，msg为空"

        # Apply chat template
        formatted_prompt = apply_chat_template(
            self.vl_processor, self.vl_config, prompt, num_images=1
        )

        output = generate(self.vl_model, self.vl_processor, formatted_prompt, image)
        result = output.text
        print(f"VL识别对方消息结果：{result}")
        print(f"📅 VL识别对方消息结束：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        result_json = json.loads(result)
        if result_json["msg"]:
            return result_json["msg"]
        else:
            raise Exception("message is none")