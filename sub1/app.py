import pickle
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter


SLACK_TOKEN = "xoxb-732147613079-731598421525-ZFaN9ODMgK1G5CjgEgQDg2oB"
SLACK_SIGNING_SECRET = "82c745be0a97984513f9d7ad407b6e37"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req. 2-1-1. pickle로 저장된 model.clf 파일 불러오기
with open("model.clf", "rb") as f : data = pickle.load(f)
##print (data)


# Req. 2-1-2. 입력 받은 광고비 데이터에 따른 예상 판매량을 출력하는 lin_pred() 함수 구현
def lin_pred(test_str):
    beta_0 = data.coef_[0]
    beta_1 = data.coef_[1]
    beta_2 = data.coef_[2]
    beta_3 = data.intercept_
    ts = test_str.split(' ')
    print(test_str)
    return (float(ts[1])*beta_0 + float(ts[2])*beta_1 + float(ts[3])*beta_2 + beta_3)

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]

    keywords = lin_pred(text)
    slack_web_client.chat_postMessage(
        channel=channel,
        text=keywords
    )

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
