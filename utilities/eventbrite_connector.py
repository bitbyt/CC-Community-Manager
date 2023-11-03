import os
from dotenv import load_dotenv
import json

import requests

load_dotenv()
browseai_api_key = os.getenv("BROWSEAI_API_KEY")
robot_id = "9be9b9f2-5ceb-42e1-aeb6-b33813796d53"
task_id = "142e0ffb-d53c-45fe-b57e-6d19a66a4bfe"
# org_id = os.getenv("EVENTBRITE_ORG_ID")

# Retrieve all events from organisation
def get_events_list():
    url = f"https://api.browse.ai/v2/robots/{robot_id}/tasks/{task_id}"

    headers = {
        'Authorization': f'Bearer {browseai_api_key}'
        }
    
    response = requests.request("GET", url, headers=headers)

    data = response.json()

    return data["result"]["capturedLists"]["Events"]

