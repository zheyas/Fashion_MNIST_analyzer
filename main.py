from subprocess import Popen, DEVNULL
from tkinter import Tk, Button, filedialog
from tkinter.messagebox import showinfo, showwarning
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

# Проверяем наличие файла модели
model_file = 'fashion_mnist_dense.h5'
if not os.path.exists(model_file):
    showwarning("Упс...", f"Файл {model_file} не найден.")
    exit()

model = load_model(model_file)
file_path = ""


def predict_digit(a):
    img_path = a
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    # Преобразуем картинку в массив
    x = image.img_to_array(img)
    # Меняем форму массива в плоский вектор
    x = x.reshape(1, 784)
    # Инвертируем изображение
    x = 255 - x
    # Нормализуем изображение
    x /= 255
    prediction = model.predict(x)
    prediction = np.argmax(prediction)

    return classes[prediction]


def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename()


def open_paint():
    try:
        Popen(["mspaint.exe"], stdout=DEVNULL, stderr=DEVNULL)
    except FileNotFoundError:
        showwarning("Упс...", "Для выполнения этой функции требуется приложение Paint.")


def show_rez():
    global file_path
    if not file_path:  # Проверяем, выбран ли файл
        showwarning("Упс...", "Пожалуйста, выберите файл для распознавания.")
        return

    if os.path.getsize(file_path) == 0:
        showwarning("Упс...", "К сожалению, этот файл пуст.")
        return

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        try:
            showinfo("Решение", "Это " + str(predict_digit(file_path)))
        except FileNotFoundError:
            showwarning("Упс...", "Для выполнения этой функции требуется приложение Paint.")
            return

    else:
        showwarning("Упс...", "К сожалению, этот файл не подходит по расширению")


root = Tk()
root['bg'] = 'pink'  # устанавливаем розовый цвет фона
root.title('Распознаем файлы')
root.geometry('250x250')  # устанавливаем размеры 300x300
root.resizable(width=False, height=False)

open_button = Button(root, text='Выбрать файл', bg='white', command=open_file_dialog)
open_button.pack(pady=(10, 0), ipadx=10)

recognize_button = Button(root, text='Распознать изображение', bg='white', command=show_rez)
recognize_button.pack(pady=(5, 0), ipadx=10)

paint_button = Button(root, text='Создать изображение', bg='white', command=open_paint)
paint_button.pack(pady=(5, 0), ipadx=10)

root.mainloop()
