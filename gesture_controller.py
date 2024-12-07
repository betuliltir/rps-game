import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import threading
import logging
import random

logging.basicConfig(level=logging.INFO)

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
        self.last_hand_position = None
        self.last_scroll_y = None

    def calculate_distance(self, p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    def detect_pinch(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Parmak uçları arasındaki mesafeyi hesapla
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Mesafe belirli bir eşiğin altındaysa pinch olarak kabul et
        return distance < 0.05

    
    def detect_two_finger_scroll(self, hand_landmarks):
        # İşaret ve orta parmak uçlarının pozisyonlarını al
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Parmak PIP eklemlerinin pozisyonlarını al
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        # İki parmağın uzatılmış olduğunu kontrol et
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        
        if index_extended and middle_extended:
            # Parmakların ortalama Y pozisyonu
            avg_y = (index_tip.y + middle_tip.y) / 2
            
            # Son pozisyonla karşılaştırarak hareket yönünü belirle
            if hasattr(self, 'last_scroll_y'):
                # Minimum hareket eşiği ekle (0.01 değerini ayarlayabilirsiniz)
                movement_threshold = 0.01
                y_difference = avg_y - self.last_scroll_y
                
                if abs(y_difference) > movement_threshold:
                    scroll_direction = 'up' if y_difference < 0 else 'down'
                    self.last_scroll_y = avg_y
                    return scroll_direction
            else:
                self.last_scroll_y = avg_y
        else:
            self.last_scroll_y = None
        
        return None

    async def handler(self, websocket):
        logging.info("New client connected")
        self.websocket = websocket
        try:
            await websocket.send(json.dumps({"status": "connected"}))
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'gameStart':
                        self.game_active = True
                        await asyncio.sleep(3)
                        if self.current_gesture:
                            game_result = self.play_game(self.current_gesture)
                            await websocket.send(json.dumps(game_result))
                        self.game_active = False
                    elif data.get('type') == 'reset':
                        self.game_active = False
                        self.current_gesture = None
                except Exception as e:
                    logging.error(f"Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected")
        finally:
            self.websocket = None

    def detect_gesture(self, hand_landmarks):
        fingers_extended = []
        
        # Thumb
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        fingers_extended.append(thumb_tip.x < thumb_ip.x)

        # Other fingers
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        for finger_tip, finger_pip in zip(finger_tips, finger_pips):
            tip = hand_landmarks.landmark[finger_tip]
            pip = hand_landmarks.landmark[finger_pip]
            fingers_extended.append(tip.y < pip.y)

        extended_count = sum(fingers_extended)

        if extended_count <= 1:
            return "rock"
        elif extended_count >= 4:
            return "paper"
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return "scissors"
        
        return "Waiting..."

    def get_hand_position(self, hand_landmarks):
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        is_pinching = self.detect_pinch(hand_landmarks)
        
        return {
            'x': index_finger_tip.x,
            'web_x': 1.0 - index_finger_tip.x,
            'y': index_finger_tip.y,
            'is_clicking': is_pinching
        }


    async def send_hand_data(self, data):
        if self.websocket:
            try:
                hand_position = data['hand_position']
                scroll_direction = data.get('scroll_direction')
                
                # Web tarafında kullanmak üzere el pozisyonunu da gönder
                await self.websocket.send(json.dumps({
                    'type': 'handPosition',
                    'x': hand_position['web_x'],
                    'y': hand_position['y'],
                    'isClicking': hand_position['is_clicking'],
                    'scrollDirection': scroll_direction
                }))
                self.last_hand_position = hand_position
            except Exception as e:
                logging.error(f"Error sending hand data: {e}")

    async def send_gesture_data(self, data):
        if self.websocket and not self.game_active:
            try:
                if data.get('gesture') != self.current_gesture:
                    await self.websocket.send(json.dumps({**data, 'type': 'gestureUpdate'}))
                    self.current_gesture = data.get('gesture')
            except Exception as e:
                logging.error(f"Error sending gesture data: {e}")

    def play_game(self, player_gesture):
        choices = ['rock', 'paper', 'scissors']
        computer_choice = random.choice(choices)
        
        if player_gesture == computer_choice:
            result = 'Tie'
        elif (player_gesture == 'rock' and computer_choice == 'scissors') or \
             (player_gesture == 'paper' and computer_choice == 'rock') or \
             (player_gesture == 'scissors' and computer_choice == 'paper'):
            result = 'Win'
        else:
            result = 'Lose'

        return {
            'type': 'gameResult',
            'result': result,
            'playerMove': player_gesture,
            'computerMove': computer_choice
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

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_data = {"gesture": "Waiting..."}
        hand_position = None
        scroll_direction = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                hand_position = self.get_hand_position(hand_landmarks)
                scroll_direction = self.detect_two_finger_scroll(hand_landmarks)
                
                # İşaret parmağı pozisyonunu ve click durumunu görsel olarak işaretle
                if hand_position:
                    h, w, _ = frame.shape
                    cx = int(hand_position['x'] * w)
                    cy = int(hand_position['y'] * h)
                    
                    # Click durumuna göre görsel geri bildirim
                    if hand_position['is_clicking']:
                        cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)  # Kırmızı dolu daire
                    else:
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)   # Yeşil nokta
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)   # Yeşil dış çember
                
                # Scroll yönünü ekrana çiz
                if scroll_direction:
                    text = f"Scroll: {scroll_direction}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                gesture = self.detect_gesture(hand_landmarks)
                if gesture:
                    gesture_data["gesture"] = gesture

            if hand_position and self.websocket:
                data_to_send = {
                    'hand_position': hand_position,
                    'scroll_direction': scroll_direction
                }
                asyncio.run_coroutine_threadsafe(
                    self.send_hand_data(data_to_send),
                    self.loop
                )

        return frame, gesture_data, hand_position

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

                frame, gesture_data, hand_position = self.process_frame(frame)

                asyncio.run_coroutine_threadsafe(
                    self.send_gesture_data(gesture_data),
                    self.loop
                )

                cv2.imshow('Rock Paper Scissors', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu ile çık
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = RPSGestureController()
    controller.start()