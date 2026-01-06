from mlx_lm import load, generate

class LMProcessor:

    def __init__(self):
        prompt = "你现在是聊天工具，根据用户给的消息，进行回答，只需直接输出答案，无需解释过程"
        self.model, self.tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
        # 对话历史
        self.messages = [{"role": "system", "content": prompt}]

    def chat(self,msg):

        if self.tokenizer.chat_template is not None:
            # 加入对话历史
            self.messages.append({"role": "user", "content": msg})
            prompt = self.tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True
            )

        response = generate(self.model, self.tokenizer, prompt=prompt, verbose=True)
        self.messages.append({"role": "assistant", "content": response})
        return response

if __name__ == '__main__':
    lm = LMProcessor()
    str = lm.chat("我想吃饭")
    print("ddd:"+str)
