# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import fasttext as ft

ft_model = ft.load_model("/home/nkosikhona/Downloads/lid.176.bin")

def language_predict(text, model=ft_model):
    text = text.replace('\n'," ")
    prediction = model.predict([text])
    return prediction
if __name__ == '__main__':
    af_text ="Opvoeding sal gemik wees op die volle ontwikkeling van die menslike persoonlikheid en op die bevordering van respek vir menseregte en fundamentele vryheid. Dit sal begrip, verdraag-saamheid en vriendskap tussen alle nasies, rasse of etniese groepe bevorder, asook die aktiwiteite van die Verenigde Volke in die handhawing van vrede."
    eng_text ="Read the following text very carefully and see what you can understand without looking at the English translation, and see what you understood from it, you can use our"
    print(language_predict(eng_text,ft_model))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
