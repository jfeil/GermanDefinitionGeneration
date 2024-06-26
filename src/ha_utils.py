import requests
import json
import enum
import os

from transformers import TrainerCallback

if 'HASSIO_BEARER' in os.environ:
    BEARER_TOKEN = os.environ['HASSIO_BEARER']
else:
    BEARER_TOKEN = ''
    print("No hassio token present, will not work!")

class Input(enum.Enum):
    TOTAL_INPUT = "input_number.ma_total_experiment"
    SINGLE_INPUT = "input_number.ma_single_experiment"
    LOSS = "input_number.ma_loss"


def _send_message(value, input_type):
    if BEARER_TOKEN == '':
        return
    url = "http://192.168.178.100:8123/api/services/input_number/set_value"
    payload = {"entity_id": input_type.value,
              "value": value}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + BEARER_TOKEN
    }
    try:
        requests.post(url, headers=headers, data=json.dumps(payload))
    except:
        pass


def set_sensor_state(current_val, max_value, input_type: Input = Input.SINGLE_INPUT):
    _send_message(((current_val) *10000 // max_value) / 100.0, input_type)

def set_absolute_value(current_val, input_type: Input = Input.SINGLE_INPUT):
    _send_message(current_val, input_type)


class HassioCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        steps_per_epoch = state.max_steps / args.num_train_epochs

        set_sensor_state(state.global_step % steps_per_epoch, steps_per_epoch, Input.SINGLE_INPUT)
        set_sensor_state(state.epoch, args.num_train_epochs, Input.TOTAL_INPUT)
        if state.log_history and 'loss' in state.log_history[-1]:
            set_absolute_value(state.log_history[-1]['loss'], Input.LOSS)

        return control
