import requests
import json
import enum
import os


if 'HASSIO_BEARER' in os.environ:
    BEARER_TOKEN = os.environ['HASSIO_BEARER']
else:
    BEARER_TOKEN = ''
    print("No hassio token present, will not work!")

class Input(enum.Enum):
    TOTAL_INPUT = "input_number.ma_total_experiment"
    SINGLE_INPUT = "input_number.ma_single_experiment"


def set_sensor_state(current_val, max_value, input_type: Input = Input.SINGLE_INPUT):
    if BEARER_TOKEN == '':
        return
    url = "http://192.168.178.100:8123/api/services/input_number/set_value"
    payload = {"entity_id": input_type.value,
              "value": ((current_val+1) *10000 // max_value) / 100.0}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + BEARER_TOKEN
    }
    requests.post(url, headers=headers, data=json.dumps(payload))