import cv2
import os
import face_recognition
import numpy as np
import telebot
from telebot import types
import threading
from constants import admin, TOKEN
from unknown_faces_handler import unknown_faces_saver, unknown_faces_sender, markup_inline_name_to_unknown

# Замените 'YOUR_TOKEN' на ваш токен, полученный от BotFather
bot = telebot.TeleBot(TOKEN)


# Define a function to run the bot's polling mechanism in a separate thread
def run_bot_polling():
    bot.polling()


# Start the bot polling in a separate thread
bot_thread = threading.Thread(target=run_bot_polling)
bot_thread.start()


def reg_name(message):
    global name
    name = message.text

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item_yes = types.KeyboardButton("Да")
    item_no = types.KeyboardButton('Нет')
    markup.add(item_yes, item_no)

    bot.send_message(admin, f'Имя нового студента -  {message.text}, верно?', reply_markup=markup)


menu = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
command1 = types.KeyboardButton("/stats")
command2 = types.KeyboardButton("/command2")
command3 = types.KeyboardButton("/command3")
command4 = types.KeyboardButton("/command4")
menu.add(command1, command2, command3, command4)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(admin, "Select a command", reply_markup=menu)


# when user starts a bot, one is given an option which statistics to get
@bot.message_handler(commands=['stats'])
def stats(message):
    user = message.from_user
    bot.reply_to(message, f"Привет, {user.first_name}! Я бот, который будет предоставлять вам статистику о "
                          f"посещаемости учеников.")

    # adding inline buttons
    markup_inline = types.InlineKeyboardMarkup(row_width=1)
    item_number_of_students = types.InlineKeyboardButton(text='Количество учеников', callback_data='number_students')
    item_number_of_unknown_faces = types.InlineKeyboardButton(text='Количество неизвестных лиц',
                                                              callback_data='number_unknowns')
    item_unknown_faces = types.InlineKeyboardButton(text="Нераспознанные лица", callback_data="unknown_faces")
    markup_inline.add(item_number_of_students, item_number_of_unknown_faces, item_unknown_faces)
    bot.send_message(message.chat.id, "Что вы хотите узнать?", reply_markup=markup_inline)


@bot.message_handler(content_types=['text'])
def check(message):
    global add_student
    if add_student and message.text != "Нет" and message.text != "Да":
        print("bob")
        reg_name(message)
    if message.text == "Да":
        markup_remove = types.ReplyKeyboardRemove()
        bot.send_message(admin, 'ученик добавлен', reply_markup=markup_remove)
        add_student = False
    elif message.text == "Нет":
        bot.send_message(admin, "напишите имя человека")


# responding to clicking on buttons
@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    global add_student
    if call.data == 'number_students':
        bot.send_message(admin, f'{num_faces} faces detected')
    elif call.data == 'number_unknowns':
        bot.send_message(admin, f'{len(unknown_face_encodings)} unknown faces are registered')
    elif call.data == 'unknown_faces':
        bot.send_message(admin, "Here is the photos of unknown faces detected:")
        for k in range(len(unknown_face_encodings)):
            with open(f'../unknown_faces/unknown_face_{k}.jpg', 'rb') as photo:
                bot.send_photo(admin, photo)
            bot.send_message(admin, 'Unknown face detected', reply_markup=markup_inline_name_to_unknown)
    elif call.data == "add_student":
        bot.send_message(admin, "напишите имя человека")
        print(call.id)
        # known_face_encodings.append()
        add_student = True


add_student = False
num_faces = 0

# Load known face encodings and names
known_face_encodings = np.load('face id v3.1/known_face_encodings.npy')
known_face_names = np.load('face id v3.1/known_face_names.npy')

unknown_face_encodings = []

# Create a directory to save the u  nknown face images
os.makedirs('unknown_faces', exist_ok=True)

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    num_faces = len(face_locations)

    i = 0
    for face_encoding in face_encodings:

        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        unknown_face_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding, tolerance=0.7)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle and label on the frame
        top, right, bottom, left = face_locations[i]

        if (name == "BOBUR" or name == "Unknown") and not (True in unknown_face_matches):
            face_image = frame[top:bottom, left:right]
            unknown_face_counter = len(unknown_face_encodings)

            unknown_faces_saver(face_image, unknown_face_counter)
            unknown_faces_sender(unknown_face_counter, bot)

            unknown_face_encodings.append(face_encoding)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        i += 1

    # Display the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
