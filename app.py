from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import pytesseract
from kivymd.uix.label import MDLabel
import numpy as np
import pandas as pd
#from PIL import Image
import tensorflow
from keras.src.models.model import model_from_json
from keras.api.preprocessing import image
model = model_from_json(open('/home/miki/Desktop/Miki/DeepLearning/NotEPy/model/model.json', 'r').read())
model.load_weights('/home/miki/Desktop/Miki/DeepLearning/NotEPy/model/model.weights.h5')

df = pd.read_csv('/home/miki/Desktop/Miki/DeepLearning/NotEPy/data/labels.csv')
df_test = df[['xmin', 'ymin', 'xmax', 'ymax']]
class MainApp(MDApp):

    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        self.label = MDLabel()
        layout.add_widget(self.image)
        layout.add_widget(self.label)
        #self.note_facade = cv2.CascadeClassifier(df_test)
        self.save_img_button = MDRaisedButton(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        self.save_img_button.bind(on_press=self.take_picture)
        layout.add_widget(self.save_img_button)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout

    def load_video(self, *args):
        labels = ('10','100','200','5','50')
        ret, frame = self.capture.read()
        if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come here
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Frame initializes
            self.image_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            note = self.note_facade.detectMultiScale(gray, 1.5, 4)
            for (x, y, w, h) in note:
                cv2.rectangle(gray, (x,y), (x+w, y+h), (255,0,0), 3)
                detected_note = self.image_frame[int(y):int(y+h), int(x):int(x+w)]
                detected_note= cv2.resize(detected_note, (35, 35))
                img_pixels = image.img_to_array(detected_note)
                img_pixels = np.expand_dims(img_pixels, axis= 0)

                img_pixels /= 255
                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                notes = labels[max_index]
                print(notes)
                cv2.putText(gray, notes, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def take_picture(self, *args):
        image_name = "/home/miki/Desktop/Miki/DeepLearning/NotEPy/data/inputs/images/DSC03240.JPG"
        img = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        text_data = pytesseract.image_to_string(img, lang='eng', config="--oem 3 --psm 6")
        print(text_data)
        self.label.text = text_data
        cv2.imshow("cv2 final image", img)
        cv2.imwrite(image_name, self.image_frame)


if __name__ == '__main__':
    MainApp().run()