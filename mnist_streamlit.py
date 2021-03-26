import streamlit as st
from architecture import FFNN, CNN
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from types import SimpleNamespace

logo = Image.open(os.path.join('img', 'logo.png'))

st.image(logo, width=200)

st.title("KI entwickeln")
st.write("### Wir werden nun gemeinsam Schritt für Schritt eine **KI entwickeln**. Genauer gesagt werden wir **Maschinelles Lernen** betreiben. "
         "Die Aufgabe der KI wird es sein, "
         "**handgeschriebene Zahlen zu erkennen**. Das heißt, wenn wir dem Computer ein Bild einer Zahl zeigen, "
         "soll er sagen, welche Zahl abgebildet ist. Das klingt vielleicht für manche einfach, für einen Computer ist "
         "das jedoch gar nicht so leicht.")

st.write("### Um eine KI zu trainieren, braucht man zuallererst **viele Daten**. In unserem Fall brauchen wir "
         "**Fotos von handgeschriebenen Zahlen**. Glücklicherweise gibt es davon schon sehr viele im Internet. "
         "Im ersten Schritt werden wir die Daten, also die Bilder, laden und schauen, wie viele es sind.")

st.write("## Daten Laden")

x_train = np.load(os.path.join("data_small", "x_test.npy"))
y_train = np.load(os.path.join("data_small", "y_test.npy"))

# Here we collect the hyperparameters we are going to use
args = SimpleNamespace(batch_size=64, test_batch_size=1000, epochs=1,
                       lr=0.01, momentum=0.5, seed=1, log_interval=100)
torch.manual_seed(args.seed)
device = 'cpu'

load_data = st.checkbox(label="Bild-Daten laden")

if load_data:
    st.write("### Du hast die Bild Daten erfolgreich geladen!")
    st.markdown("### **Zusammenfassung:**")
    st.write(f"### Die Daten bestehen aus **70,000 Bildern**!")
    st.write(f"### Jedes Bild ist **28 x 28 Pixel** groß.")
st.write(" ")

if load_data:
    b1 = st.checkbox("Nächster Schritt: Bilder abbilden")

if load_data and b1:
    st.write("## Bilder abbilden")
    st.write("### Als nächstes möchtest du dir sicher ein paar Beispiel Bilder ansehen!")

    n_img = st.selectbox("Wähle aus, wie viele Bilder du ansehen möchtest:", options=[0, 10, 15, 20, 30])
    rows = int(n_img/5)

if load_data and b1 and n_img != 0:
    fig, ax = plt.subplots(rows, 5)
    fig.set_figheight(1+1.2*rows)
    fig.suptitle("Beispiel Bilder")
    i = 0
    for x in range(rows):
        for y in range(5):
            two_d = (np.reshape(x_train[i], (28, 28)) * 255).astype(np.uint64)
            ax[x, y].imshow(two_d, cmap='gray_r')
            ax[x, y].set_title(f"{y_train[i]}")
            ax[x, y].set_xticks([])
            ax[x, y].set_yticks([])
            # ax[x,y].axis('off')
            i += 1
    st.pyplot(fig)
    st.write("### Jedes Bild hat eine sogenannte **Klasse**. In unserem Fall ist das **die Zahl, die auf dem Bild zu sehen ist**. "
             "Die Klasse jedes Bildes ist darüber abgebildet. Die Aufgabe unserer KI wird es später sein, genau diese"
             " Klasse, also welche Zahl zu sehen ist, herauszufinden.")

st.write(" ")

#if load_data and b1:
    #b2 = st.checkbox("Nächster Schritt: Aufgabe")
b2 = True
st.write(" ")

if load_data and b1 and b2 and False:
    idx = 50
    st.write("## Aufgabe")
    two_d = (np.reshape(x_train[idx], (28, 28)) * 255).astype(np.uint64)
    two_d_l = y_train[idx]
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    fig.set_figheight(2)

    st.write("### Welche Nummer ist unten abgebildet?")
    answer = st.selectbox(label="Wähle eine Nummer aus:", options=[i for i in range(10)])
    if answer == two_d_l:
        plt.title(f"{two_d_l}")
        st.write("### Das ist **korrekt**!")
    else:
        st.write("### Das ist noch **nicht** richtig!")

    plt.imshow(two_d, cmap='gray_r')
    st.pyplot(fig)

if load_data and b1 and b2 and False:
    st.write("### Für uns **Menschen** ist diese **Aufgabe sehr einfach**. Für einen **Computer jedoch nicht ganz so leicht**. Dieser hat "
             "kein Verständnis wie wir Mesnchen und muss sich Pixel für Pixel des Bildes ansehen um zu wissen, welche Zahl"
             " abgebildet ist. Wir werden in den nächsten Schritten nun gemeinsam eine KI"
             " trainieren welche "
             "handgeschriebenen Zahlen benennen kann. Los gehts!")

if load_data and b1 and b2:
    b3 = st.checkbox("Nächster Schritt: Modell auswählen")

st.write(" ")

input_dim = 784
output_dim = 10

if load_data and b1 and b2 and b3:
    st.write("## Modell auswählen")
    st.write("### Eine KI benötigt einen **Bauplan**, wie sie aufgebaut sein soll. Dafür gibt es **verschiedene Modelle**. "
             "Diese unterscheiden sich darin, wie sie ein Bild verarbeiten und zu einer Entscheidung kommen. "
             "Wir werden ein **Künstliches Neuronales Netzwerk** verwenden. Davon haben wir ja schon gehört. "
             "Wähle im nächsten Schritt das Neuronale Netz aus.")
    clf_name = st.selectbox(label="Wähle eine Modell aus", options=["Nichts", "Neuronales Netz"])
    if clf_name == "Neuronales Netz":
        accuracy = 93.30
        model = FFNN(input_dim, output_dim).to(device)
        model.load_state_dict(torch.load(os.path.join('model', 'ffnn')))
        st.write("## Neuronales Netzwerk")
        nn = Image.open(os.path.join('img', 'nn.PNG'))
        st.image(nn, caption="Neuronales Netzwerk", use_column_width=True)
        st.write("### Das ist ein **Bild eines Künstlichen Neuronalen Netzwerks**. Diese heißen so, weil sie so ähnlich funktionieren "
                 "wie die **Neuronen in unserem Gehirn**. Neuronen sind unsere Gehirnzellen und diese leiten Informationen "
                 "in unserem Körper weiter. Das gleiche passiert auch hier. Ganz **links** werden die **Pixel der Bilder "
                 "aufgenommen**. In der **Mitte** wird das **Bild verarbeitet** und **rechts spuckt das Neuronale Netz dann "
                 "aus, welche Zahl zu sehen ist**.")
    if clf_name == "CNN":
        accuracy = 95.60
        model = model = CNN(use_batch_norm=True, n_blocks=3, n_layers=1, channels=16, multiply_channels=1,
                            global_max=False).to(device)
        model.load_state_dict(torch.load(os.path.join('model', 'cnn')))
        st.write("## CNN")
        st.write("### CNN ist eine Abkürzung und steht für \"Konvolutionales Neuronales Netz\". Hier steckt im "
                 "Namen auch ein Neuronales Netz drinnen. Diese Art ist aber schon etwas komplizierter, "
                 "also werden wir dieses nicht genauer erklären. Merk dir einfach, dass CNNs vor allem "
                 "für die Klassifizierung von Bildern sehr gut geeignet sind.")

st.write(" ")
if load_data and b1 and b2 and b3:
    b4 = st.checkbox("Nächster Schritt: Modell trainieren")

if load_data and b1 and b2 and b3 and b4:
    st.write("## Modell trainieren")
    st.write("### Jetzt **trainieren wir unser Modell**. Dabei zeigen wir dem Neuronalen Netzwerk viele Beispiel Bilder. "
             "**Mit jedem Bild lernt es dazu** und wird immer besser, bis es fast alle Zahlen richtig benennen kann.")

    st.write("### Drücke den Knopf unten, um deine KI zu trainieren.")
    training = st.button(label="KI trainieren")
    if training:
        st.write("#### Du hast deine KI erfolgreich trainiert!")
    st.write("### Hier passiert nun sehr viel im Hintergrund. Merk dir einfach, dass deine KI mit den Bildern, die wir "
             "oben gesehen haben, trainiert wurde und diese nun benennen kann. **Die KI macht aber manchmal Fehler**. Sehen wir uns nun an, wie gut diese ist.")

st.write(" ")
if load_data and b1 and b2 and b3 and b4:
    b5 = st.checkbox("Nächster Schritt: Wie gut ist deine KI?")

if load_data and b1 and b2 and b3 and b4 and b5:
    st.write("## Wie gut ist deine KI?")
    st.write("### Im letzten Schritt werden wir uns ansehen, wie gut unsere KI ist und werden uns ein paar Ergebnisse "
             "ansehen.")
    st.write("### Als erstes wollen wir uns die **Genauigkeit unserer KI** ansehen. Die Genauigkeit ist ein **Prozentwert** "
             "und sagt uns, in wie viel Prozent der Fälle die KI eine Zahl richtig identifizieren kann. **Je höher dieser "
             "Wert ist, desto besser**.")

    acc = st.checkbox(label="Genaugkeit")
    if clf_name == "Nichts":
        st.write("### **Du musst oben zuerst ein Modell auswählen!**")
    elif acc:
        st.write(f"### Deine KI hat eine **Genauigkeit** von **{accuracy}%**!")
        st.write(f"### Das heißt, deine KI würde im Durchschnitt **von 100 Bildern** ungefähr **{int(round(accuracy))}** richtig erkennen!")

st.write(" ")
if load_data and b1 and b2 and b3 and b4 and b5:
    b6 = st.checkbox("Nächster Schritt: KI testen")

if load_data and b1 and b2 and b3 and b4 and b5 and b6:
    st.write("### Nun schauen wir uns ein paar **Beispiele** an. **Wähle mit dem Slider eine Zahl aus**. Danach kannst du dir "
             "ansehen, ob deine KI diese Zahl richtig erkennt oder nicht.")
    st.write("#### Schau dir die Nummer 8 am Schieberegler genauer an. Was fällt dir auf?")
    slider = st.slider(label="Wähle mit dem Schieberegler eine Zahl aus,", min_value=0, max_value=99)

    if clf_name == "Neuronales Netz":
        two_d = (np.reshape(x_train[slider], (28, 28)) * 255).astype(np.float32)
        two_d_l = y_train[slider]
        fig = plt.figure()
        plt.title(f"{two_d_l}")
        plt.imshow(two_d, cmap='gray_r')

        two_d = torch.tensor(two_d)
        model.eval()
        with torch.no_grad():
            two_d = Variable(two_d.view(-1, input_dim))
            two_d = torch.tensor(two_d, dtype=torch.float32)
            two_d = two_d.to(device)
            output = model(two_d)
            pred = output.max(1, keepdim=True)[1]
            pred = int(pred)

    if clf_name =="CNN":
        trans = transforms.Normalize((0.1307,), (0.3081,))
        two_d = (np.reshape(x_train[slider], (28, 28)) * 255).astype(np.float32)
        two_d_l = y_train[slider]
        fig = plt.figure()
        plt.title(f"{two_d_l}")
        plt.imshow(two_d, cmap='gray_r')

        two_d = np.reshape(x_train[slider], (28, 28)).astype(np.float32)
        two_d = torch.tensor(two_d)
        model.eval()
        with torch.no_grad():
            two_d = two_d.view(1,1,28,28)
            two_d = torch.tensor(two_d, dtype=torch.float32)
            two_d = two_d /255
            two_d = trans(two_d)
            two_d = two_d.to(device)
            output = model(two_d)
            pred = output.max(1, keepdim=True)[1]
            pred = int(pred)

    if clf_name == "Nichts":
        st.write("### **Du musst oben zuerst ein Modell auswählen!**")

    if not clf_name == "Nichts":
        st.write(f"## KI Klassifizierung: **{pred}**")
        st.write(f"## Richtige Zahl: **{two_d_l}**")
        st.pyplot(fig)
        st.write("## **Gratuliere! Du hast erfolgreich eine KI trainiert! Geh nun zum Geogebra Buch zurück.**")
