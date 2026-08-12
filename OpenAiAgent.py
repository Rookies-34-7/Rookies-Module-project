import os
import json
from openai import OpenAI


# 사용자 정의 함수
def blank():
    return 0
  

# Function Calling Schema
TOOLS = [
    {
        "type": "",
        "name": "",
        "description": "",
        "parameters": {},
    }
]



# OpenAI Agent 클래스
class OpenAIAgent:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
            )

        self.client = OpenAI(api_key=api_key)

        self.model = "gpt-5.5"

    # OpenAI API 호출
    def gen_response(self, user_message):
        
        response = self.client.responses.create(
            model=self.model,
            input=user_message,
            tools=TOOLS
        )

        return response
    
    # 함수 실행
    def handle_function_call(self, function_name, arguments):
        
        if function_name == "function":
            return 
        
        else:
            result = "지원하지 않는 함수입니다."

        return result

    
    # 전체 대화 흐름
    def chat(self, user_message):

        response = self.gen_response(user_message)

        function_outputs = []
        function_logs = []

        for item in response.output:

            # Function Calling이 발생한 경우
            if item.type == "function_call":

                function_name = item.name
                arguments = json.loads(item.arguments)

                result = self.handle_function_call(
                    function_name,
                    arguments
                )

                function_logs.append({
                    "name": function_name,
                    "arguments": arguments,
                    "result": result
                })

                function_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": str(result)
                })

        
        # 함수 호출 여부 판단

        if function_outputs:

            second_response = self.client.responses.create(
                model=self.model,
                previous_response_id=response.id,
                input=function_outputs,
                tools=TOOLS
            )

            final_answer = second_response.output_text

        else:
            final_answer = response.output_text

        return final_answer, function_logs