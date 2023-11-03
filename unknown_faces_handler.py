import cv2
from telebot import types
from constants import admin

markup_inline_name_to_unknown = types.InlineKeyboardMarkup(row_width=1)
item_add_student = types.InlineKeyboardButton(text='Добавить ученика', callback_data='add_student')
markup_inline_name_to_unknown.add(item_add_student)


def unknown_faces_saver(face_image, unknown_face_counter):
    filename = f'unknown_faces/unknown_face_{unknown_face_counter}.jpg'
    cv2.imwrite(filename, face_image)


def unknown_faces_sender(unknown_face_counter, bot):
    with open(f'unknown_faces/unknown_face_{unknown_face_counter}.jpg', 'rb') as photo:
        bot.send_photo(admin, photo)
        bot.send_message(admin, 'Unknown face detected')