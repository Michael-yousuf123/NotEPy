from kivy.core.text import LabelBase
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.core.window import Window
Window.size = (310, 580)
KV = '''
MDScreen:
    MDFloatLayout:
        md_bg_color: [1,1,1,1]
        Image:
            source: "/home/miki/Desktop/Miki/DeepLearning/NotEPy/assets/icons/android-chrome-192x192.png"
            pos_hint: {'center_x': 0.18,'center_y': 0.95}
        Image:
            source: "/home/miki/Desktop/Miki/DeepLearning/NotEPy/assets/imgs/birr-1024x574.png"
            size_hint: .8, .8
            pos_hint:{'center_x':.5, 'center_y': .65} '''
class NoteApp(MDApp):
    """
    """
    def build(self):
        return Builder.load_string(KV)
if __name__ == '__main__':
    NoteApp().run()