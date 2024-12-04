import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import threading
import logging
import random

logging.basicConfig(level=logging.DEBUG)

class RPSGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.websocket = None
        self.game_active = False
        self.current_gesture = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.start_websocket_server()

    async def handler(self, websocket):
        logging.info("New client connected")
        self.websocket = websocket
        try:
            await websocket.send(json.dumps({"status": "connected"}))
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logging.debug(f"Received message: {data}")
                    if data.get('type') == 'gameStart':
                        self.game_active = True
                        logging.info("Game started, waiting 3 seconds...")
                        await asyncio.sleep(3)
                        if self.current_gesture:
                            game_result = self.play_game(self.current_gesture)
                            logging.info(f"Game result: {game_result}")
                            await websocket.send(json.dumps(game_result))
                        else:
                            logging.warning("No gesture detected at game end")
                        self.game_active = False
                    elif data.get('type') == 'reset':
                        self.game_active = False
                        self.current_gesture = None
                        logging.info("Game reset")
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    continue
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected")
        finally:
            self.websocket = None

    def play_game(self, player_gesture):
        choices = ['rock', 'paper', 'scissors']
        computer_choice = random.choice(choices)
        
        logging.info(f"Player chose: {player_gesture}, Computer chose: {computer_choice}")

        if player_gesture == computer_choice:
            result = 'Tie'
        elif (player_gesture == 'rock' and computer_choice == 'scissors') or \
             (player_gesture == 'paper' and computer_choice == 'rock') or \
             (player_gesture == 'scissors' and computer_choice == 'paper'):
            result = 'Win'
        else:
            result = 'Lose'

        return {
            'result': result,
            'playerMove': player_gesture,
            'computerMove': computer_choice,
            'type': 'gameResult'
        }

    def start_websocket_server(self):
        async def serve():
            async with websockets.serve(self.handler, "localhost", 8765):
                logging.info("WebSocket server started on ws://localhost:8765")
                await asyncio.Future()

        def run_server():
            self.loop.run_until_complete(serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    async def send_gesture_data(self, data):
        if self.websocket and not self.game_active:
            try:
                if data.get('gesture') != self.current_gesture:
                    logging.debug(f"Sending gesture data: {data}")
                    await self.websocket.send(json.dumps({**data, 'type': 'gestureUpdate'}))
                    self.current_gesture = data.get('gesture')
            except Exception as e:
                logging.error(f"Error sending gesture data: {e}")

    def detect_gesture(self, hand_landmarks):
        fingers_extended = []

        # Thumb
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        
        # Calculate thumb state based on horizontal position
        fingers_extended.append(thumb_tip.x < thumb_ip.x)

        # Other fingers
        fingers = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        fingers_pip = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        for finger_tip, finger_pip in zip(fingers, fingers_pip):
            tip = hand_landmarks.landmark[finger_tip]
            pip = hand_landmarks.landmark[finger_pip]
            # Finger is extended if tip is higher than pip
            fingers_extended.append(tip.y < pip.y)

        extended_count = sum(fingers_extended)
        
        # Debug information
        logging.debug(f"Extended fingers: {fingers_extended}, Count: {extended_count}")

        if extended_count <= 1:
            return "rock"
        elif extended_count >= 4:
            return "paper"
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return "scissors"
        
        return "Waiting..."

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_data = {"gesture": "Waiting..."}

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks)
                if gesture:
                    gesture_data["gesture"] = gesture
                    logging.debug(f"Detected gesture: {gesture}")

        return frame, gesture_data

    def start(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return

        logging.info("Camera opened successfully")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logging.warning("Failed to read frame")
                    continue

                frame, gesture_data = self.process_frame(frame)

                asyncio.run_coroutine_threadsafe(
                    self.send_gesture_data(gesture_data),
                    self.loop
                )

                cv2.imshow('Rock Paper Scissors', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
                    logging.info("ESC pressed, exiting...")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = RPSGestureController()
    controller.start()