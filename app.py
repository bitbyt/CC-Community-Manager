import os
from dotenv import load_dotenv
import json

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from utilities.config_loader import load_persona
from utilities.chat_agents import general_response, knowledge_retrieval
from utilities.eventbrite_connector import get_events_list

load_dotenv()


BASE_PERSONA = os.getenv("PERSONA")
personaname = BASE_PERSONA.title()
persona = {}
load_persona(persona)
instructions = persona[BASE_PERSONA]

# Install the Slack app and get xoxb- token in advance
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

app = App(token=SLACK_BOT_TOKEN)

@app.command("/hello")
def hello_command(ack, body):
    user_id = body["user_id"]
    ack(f"Hi, <@{user_id}>!")

@app.command("/upcoming_events")
def hello_command(ack, respond):
    ack(f"Upcoming Calm Circles events.")
    events = get_events_list()
    # print(events)

    eventBlocks = []

    for event in events:
        block = {
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": f"*{event['Title']}*\n {event['Date']}\n <{event['Event Link']}|View Event>"
			},
			"accessory": {
				"type": "image",
				"image_url": event['Image URL'],
				"alt_text": event['Title']
			}
		}
        eventBlocks.append(block)

    respond(blocks=eventBlocks)

@app.command("/ask")
def repeat_text(ack, respond, command):
    user_id = command["user_id"]
    query = command['text']

    # Acknowledge command request
    ack(f"Question: {command['text']}")

    data = knowledge_retrieval(query)
    dataDict = json.loads(data)
    answer = dataDict["answer"]
    sources = dataDict["sources"]

    sourceBlock = {}

    if type(sources) == list:
        sources = "\n".join(sources)
        sourceBlock = {
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": f"See more: {sources}"
			}
		}
    else:
        sourceBlock = {
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "Tap on the button to learn more."
			},
			"accessory": {
				"type": "button",
				"text": {
					"type": "plain_text",
					"text": "View Page",
					"emoji": true
				},
				"value": "click_me_123",
				"url": sources,
				"action_id": "button-action"
			}
		}


    blocks = [{
        "type": "section",
        "text": {
            "type": "mrkdwn", 
            "text": dataDict["answer"]
        },
    },
    sourceBlock
    ]

    respond(blocks=blocks)

@app.event("app_mention")
def mentions(event, say, client):
    response = general_response(instructions, event, personaname)

    message_ts = event["ts"]
    channel_id = event["channel"]

    try:  
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=message_ts,
            text=response
        )

    except Exception as e:
        print(f"Error: {e}")

    # say(response)

@app.event({
    "type": "message",
    "channel_type": "im"
})
def direct_message(event, say):
    response = general_response(instructions, event)

    say(response)

# @app.event("message")
# def event_test(say, body):
#     say(f"Hi!")

if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()