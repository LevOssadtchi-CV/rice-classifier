import telebot
from telebot import types
import os
import json
import random

TOKEN = "8444981556:AAFdWZNlK6x8jERAmGclHBMbmqz6xQgTTcE"
bot = telebot.TeleBot(TOKEN, threaded=True)

BOT_IMG_DIR = "bot_img"
RICE_TEST_DIR = "Rice_split/test"
RESULTS_DIR = "results"  # Папка для пользователей
ROUNDS = 10
CATEGORIES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Загружаем справочные картинки
bot_imgs = {name.split(".")[0]: os.path.join(BOT_IMG_DIR, name) for name in os.listdir(BOT_IMG_DIR)}

# Загружаем тестовые картинки для каждого вида
test_imgs = {}
for cat in CATEGORIES:
    cat_dir = os.path.join(RICE_TEST_DIR, cat)
    test_imgs[cat] = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

# Пользовательские сессии
user_sessions = {}

# --- Сохраняем результат в файл текущей игры ---
def save_result(user_id, round_info):
    session = user_sessions[user_id]
    file_path = session.get("current_file")
    if not file_path:
        return

    # Загружаем существующие данные, добавляем раунд
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(round_info)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# === Ознакомительные фото ===
def send_intro_photo(chat_id, user_id):
    session = user_sessions[user_id]

    if session["intro_index"] >= len(CATEGORIES):
        # Все ознакомительные фото показаны, кнопка "Начать игру"
        markup = types.InlineKeyboardMarkup()
        btn = types.InlineKeyboardButton("Начать игру", callback_data="start_game")
        markup.add(btn)
        msg = bot.send_message(chat_id, "Вы просмотрели все справочные фото.", reply_markup=markup)
        session["last_intro_msg_id"] = msg.message_id
        return

    # Отправляем следующее фото с кнопкой "Запомнил"
    cat = CATEGORIES[session["intro_index"]]
    markup = types.InlineKeyboardMarkup()
    btn = types.InlineKeyboardButton("Запомнил", callback_data="next_intro")
    markup.add(btn)
    msg = bot.send_photo(chat_id, photo=open(bot_imgs[cat], "rb"), caption=cat, reply_markup=markup)
    session["last_intro_msg_id"] = msg.message_id

# === Игра ===
def send_next_round(chat_id, user_id):
    session = user_sessions[user_id]
    if session["round"] >= ROUNDS:
        # Игра окончена, показать кнопку "Начать игру" снова
        markup = types.InlineKeyboardMarkup()
        btn = types.InlineKeyboardButton("Начать игру", callback_data="start_game")
        markup.add(btn)
        msg = bot.send_message(chat_id, "Игра окончена! Хотите сыграть еще раз?", reply_markup=markup)
        session["last_intro_msg_id"] = msg.message_id
        return

    chosen_cat = random.choice(CATEGORIES)
    img_path = random.choice(test_imgs[chosen_cat])
    session["current_cat"] = chosen_cat
    session["current_img"] = img_path
    session["round"] += 1

    # Кнопки выбора
    markup = types.InlineKeyboardMarkup()
    buttons = [types.InlineKeyboardButton(name, callback_data=name) for name in CATEGORIES]
    markup.add(*buttons)

    msg = bot.send_photo(chat_id, photo=open(img_path, "rb"),
                         caption=f"Раунд {session['round']}: Угадай сорт",
                         reply_markup=markup)
    session["last_msg_id"] = msg.message_id

# === Старт ===
@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Инициализация сессии
    user_sessions[user_id] = {
        "intro_index": 0,
        "round": 0,
        "shown": [],
        "last_msg_id": None,
        "last_intro_msg_id": None,
        "current_file": None
    }

    # Приветствие с кнопкой "Готов?"
    markup = types.InlineKeyboardMarkup()
    btn = types.InlineKeyboardButton("Готов?", callback_data="ready_intro")
    markup.add(btn)
    bot.send_message(chat_id, "Привет! Давай сначала познакомимся с сортами риса. Готов?", reply_markup=markup)

# === Обработка кнопок ===
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    user_id = call.from_user.id
    chat_id = call.message.chat.id
    session = user_sessions[user_id]

    if call.data == "ready_intro" or call.data == "next_intro":
        # Удаляем только кнопку (оставляем фото)
        try:
            bot.edit_message_reply_markup(chat_id, session["last_intro_msg_id"], reply_markup=None)
        except:
            pass

        if call.data == "next_intro":
            session["intro_index"] += 1
        send_intro_photo(chat_id, user_id)

    elif call.data == "start_game":
        # Удаляем кнопку "Начать игру"
        try:
            bot.edit_message_reply_markup(chat_id, session["last_intro_msg_id"], reply_markup=None)
        except:
            pass

        session["round"] = 0

        # Создаем новый файл для этой игры
        user_folder = os.path.join(RESULTS_DIR, str(user_id))
        os.makedirs(user_folder, exist_ok=True)
        existing_files = [f for f in os.listdir(user_folder) if f.endswith(".json")]
        new_index = len(existing_files) + 1
        file_path = os.path.join(user_folder, f"{new_index}.json")
        session["current_file"] = file_path

        send_next_round(chat_id, user_id)

    elif call.data in CATEGORIES:
        # Проверяем ответ
        user_choice = call.data
        correct = user_choice == session["current_cat"]

        round_info = {
            "round": session["round"],
            "shown_image": os.path.basename(session["current_img"]),
            "chosen": user_choice,
            "correct": correct
        }
        save_result(user_id, round_info)

        # Удаляем только игровое сообщение
        try:
            bot.delete_message(chat_id, session["last_msg_id"])
        except:
            pass

        send_next_round(chat_id, user_id)

# === Запуск бота ===
print("Bot is running...")
bot.infinity_polling()
