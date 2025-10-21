from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')

myFlag = False 

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    command = {}


    if turret_x < 90:
        command.update({ "turretQE" : {"command": 'E', "weight": 1.0}, })

    9
    if myFlag == False:
        pass

    else:
        command.update({ "moveWS" : {"command" : 'W', "weight" : 0.3}, 
                         "moveAD" : {"command" : 'A', "weight" : 0.5}})
    

    print("🔁 Sent Combined Action:", command)
    return jsonify(command)


#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",  # Options: "start" or "pause"
        "blStartX": 180,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 140,
        "rdStartX": 180, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 160,
        "trackingMode": True,
        "detactMode": False,
        "logMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)
