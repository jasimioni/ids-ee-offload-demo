#!/usr/bin/env python

import pika
import pandas as pd
import os
import uuid
import pickle
import logging
import torch
from datetime import datetime
import time
import json
import sys
import argparse
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'

rundir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rundir)
sys.path.append(os.path.dirname(rundir))

from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits
from calibration.temperature_scaling_2exits import ModelWithTemperature
from utils.functions import *

parser = argparse.ArgumentParser(description='Early Exits processor client.')

parser.add_argument('--mq-username', help='RabbitMQ username')
parser.add_argument('--mq-password', help='RabbitMQ password')
parser.add_argument('--mq-hostname', help='RabbitMQ hostname', required=True)
parser.add_argument('--mq-queue', help='RabbitMQ queue', default='ee-processor')
parser.add_argument('--device', help='PyTorch device', default='cpu')
parser.add_argument('--trained-network-file', help='Trainet network file', required=True)
parser.add_argument('--network', help='Network to use AlexNet | MobileNet', required=True)
parser.add_argument('--dataset', help='Dataset to use', required=True)
parser.add_argument('--normal-exit1-min-certainty', help='Minimum certainty for normal exit 1', default=0.95, type=float)
parser.add_argument('--attack-exit1-min-certainty', help='Minimum certainty for attack exit 1', default=0.95, type=float)
parser.add_argument('--normal-exit2-min-certainty', help='Minimum certainty for normal exit 2', default=0.98, type=float)
parser.add_argument('--attack-exit2-min-certainty', help='Minimum certainty for attack exit 2', default=0.98, type=float)
parser.add_argument('--process-exit2-locally', help='Do not offload exit2 requests', action='store_true')
parser.add_argument('--debug', help='Enable debug messages', action='store_true')

args = parser.parse_args()

log_level = logging.INFO if args.debug else logging.WARN
logging.basicConfig(level=log_level,
                    format='%(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)

device = torch.device(args.device)
if args.network == 'MobileNet':
    model = MobileNetV2WithExits().to(device)
else:
    model = AlexNetWithExits().to(device)

model_t = ModelWithTemperature(model, device=device)
model_t.load_state_dict(torch.load(args.trained_network_file, map_location=device))
model_t.model.eval()
model_t.model(torch.rand(1, 1, 8, 8).to(device))  # Run the network once to cache it

df = pd.read_csv(args.dataset)

show_accuracy = 1
if len(df.columns) == 58:
    show_accuracy = 0
    df['class'] = 0

data   = CustomDataset(as_matrix=True, df=df)
loader = DataLoader(data, batch_size=1, shuffle=True)

class EEProcessorClient(object):
    def __init__(self):
        connection_params = { 'host': args.mq_hostname }
        if args.mq_username and args.mq_password:
            credentials = pika.PlainCredentials(args.mq_username, args.mq_password)
            connection_params['credentials'] = credentials

        self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(**connection_params))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, body):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=args.mq_queue,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=pickle.dumps(body))
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        
        try:
            return pickle.loads(self.response)
        except Exception as e:
            raise Exception(f"Failed to load: {e}")


def get_times():
    return time.time(), time.process_time()

def get_elapsed_times(st_time, st_process_time):
    return time.time() - st_time, time.process_time() - st_process_time

def green(s):
    return f'\033[92m{s}\033[0m'

def red(s):
    return f'\033[91m{s}\033[0m'

def orange(s):
    return f'\033[93m{s}\033[0m'

def s_correct():
    return green('✓ Correct')

def s_incorrect():
    return red('✗ Incorrect')

def s_rejected():
    return orange('? Rejected')

eeprocessor = EEProcessorClient()

correct_total = 0
total = 0
rejected_total = 0
offloaded_total = 0
dfs = []
for b, (X, y) in enumerate(loader):
    X = X.to(device)
    y = y.to(device)
    count = len(y)
    total += count

    st_time, st_process_time = get_times()
    bb1 = model_t.model.backbone[0](X)
    e1 = model_t.model.exits[0](bb1)
    y_pred = model_t.temperature_scale(0, e1)
    
    elapsed_time, elapsed_process_time = get_elapsed_times(st_time, st_process_time)
    
    df = pd.DataFrame()

    certainty, predicted = torch.max(nn.functional.softmax(y_pred, dim=-1), 1)
    certainty = certainty.cpu().detach().numpy()
    predicted = predicted.cpu().detach().numpy()
    
    df['exit_1_certainty'] = certainty
    df['exit_1_prediction'] = predicted
    
    mask = (df['exit_1_prediction'] == 0) & (df['exit_1_certainty'] < args.normal_exit1_min_certainty) | \
           (df['exit_1_prediction'] == 1) & (df['exit_1_certainty'] < args.attack_exit1_min_certainty)   
    
    to_offload = []
    for n, val in enumerate(mask):
        if val:
            to_offload.append(bb1[n].detach().numpy())
    
    
    c = f"{100 * certainty.mean():4.2f}%"
    c = green(c) if not to_offload else red(c)
    if to_offload:
        c = f"{c} -> "
    
    print(f"Exit 1: {1e3 * elapsed_time:5.2f}ms Certainty: {c}", end='')
    
    sys.stdout.flush()
    
    if to_offload:
        offloaded = len(to_offload)
        offloaded_total += offloaded
            
        bb1 = torch.tensor(numpy.array(to_offload)).to(device)
        st_time, st_process_time = get_times()
        reqid = f"{b:04x}"
        if args.process_exit2_locally:
            bb2 = model_t.model.backbone[1](bb1)
            e2 = model_t.model.exits[1](bb2)
            y_pred_remote = model_t.temperature_scale(1, e2)
            print(f"LOCAL  -> ", end='')
        else:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            
            request = {
                'timestamp': now,
                'bb1': bb1,
                'reqid': reqid
            }
            
            # logger.warning(f" [x] Requesting {now} {offloaded} being offloaded")
            
            print(f"0x{reqid} -> ", end='')
            sys.stdout.flush()

            response = eeprocessor.call(request)
            
            y_pred_remote = response['output']
            hostname = response['hostname']
            
        elapsed_time, elapsed_process_time = get_elapsed_times(st_time, st_process_time)
        certainty_r, predicted_r = torch.max(nn.functional.softmax(y_pred_remote, dim=-1), 1)
            
        certainty_r = certainty_r.cpu().detach().numpy().tolist()
        predicted_r = predicted_r.cpu().detach().numpy().tolist()
        print(f"Exit 2: {1e3 * elapsed_time:6.2f}ms (CPU: {1e3 * elapsed_process_time:7.2f}ms)", end='')
    else:
        print('                                                                 ', end='')
        certainty_r = []
        predicted_r = []
    
    c_certainty = []
    c_predicted = []
    
    final_certainty = []
    final_predicted = []
    
    for n, val in enumerate(mask):
        if val:
            c_certainty.append(certainty_r.pop(0))
            c_predicted.append(predicted_r.pop(0))
            
            if c_predicted[-1] == 0 and c_certainty[-1] < args.normal_exit2_min_certainty or \
               c_predicted[-1] == 1 and c_certainty[-1] < args.attack_exit2_min_certainty:
                final_certainty.append('rejected')
                final_predicted.append('rejected')
            else:
                final_certainty.append(certainty[n])
                final_predicted.append(predicted[n])
        else:
            c_certainty.append('N/A')
            c_predicted.append('N/A')
            final_certainty.append(certainty[n])
            final_predicted.append(predicted[n])
    
    df['exit_2_certainty'] = c_certainty
    df['exit_2_prediction'] = c_predicted
    df['final_certainty'] = final_certainty
    df['final_prediction'] = final_predicted

    rejected = (df['final_prediction'] == 'rejected').sum()
    rejected_total += rejected
    
    # For this demo we are processing 1 event at a time, so getting the last item from list to print
    
    prediction = final_predicted[-1]
    
    if to_offload:
        c = f"{100 * c_certainty[-1]:4.2f}%"
        c = green(c) if prediction != 'rejected' else red(c)
        
        print(f" Certainty: {c}", end='')
    
    #  print(f"Rejection: {rejected/count*100:.2f}%")
    
    print(" | Prediction: ", end='')
    
    if prediction == 'rejected':
        print(s_rejected())
    else:
        if prediction == y[-1]:
            print(s_correct())
        else:
            print(s_incorrect())
                     
    if show_accuracy:
        df['class'] = y.cpu().numpy()
        correct = (df['final_prediction'] == df['class']).sum()
        correct_total += correct
    else:
        df['class'] = 'N/A'
    
    dfs.append(df)
    time.sleep(1)
    
if total == rejected_total:
    accuracy = 'N/A (all rejected)'
else:
    accuracy = f"{100 * correct_total/(total-rejected_total):.2f}%"
rejection = f"{100 * rejected_total/total:.2f}%"
r_exit1 = f"{( total - offloaded_total)/total*100:.2f}%"
r_exit2 = f"{offloaded_total/total*100:.2f}%"

print(f"""\nFinal stats:
    Accuracy: {accuracy}
    Rejection: {rejection}
    Resolved in Exit1: {r_exit1}
    Offloaded (Exit2): {r_exit2}
""")