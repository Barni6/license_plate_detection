import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk
import imutils

class Rendszamolvaso:
    def __init__(self, root):
        # Inicializálja a GUI-t
        self.root = root
        self.root.title("Rendszámolvasó")
        self.root.geometry("1600x900")

        # Képek tárolására szolgáló változók
        self.loaded_image = None
        self.processed_image = None

        # Gombokat és feliratokat tartalmazó keret inicializálása
        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=0, column=0, sticky=tk.NW, padx=10, pady=10)

        # "Kép betöltése" gomb
        self.load_button = tk.Button(self.button_frame, text="Kép betöltése", command=self.load_image, bg="beige", fg="black")
        self.load_button.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)

        # "Előfeldolgozás és rendszám szövegének kiírása" gomb
        self.process_button = tk.Button(self.button_frame, text="Előfeldolgozás és \nrendszám szövegének kiírása", command=self.display_processed_licensePlate_image, state=tk.DISABLED, bg="beige", fg="black")
        self.process_button.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)

        # Kép megjelenítésére szolgáló címke
        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NW)

        # Eredmény megjelenítésére szolgáló címke
        self.result_label = tk.Label(self.button_frame, text="")
        self.result_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

    def load_image(self):
        # Kép betöltése
        file_path = filedialog.askopenfilename()
        if file_path:
            self.loaded_image = cv2.imread(file_path)
            if self.loaded_image is not None:
                # Sikeres képbetöltés esetén megjeleníti a képet és engedélyezi a feldolgozás gombot
                self.display_image(self.loaded_image)
                self.process_button.config(state=tk.NORMAL)
            else:
                print("A kép beolvasása sikertelen.")
        else:
            print("Nincs kiválasztott fájl.")

    def display_image(self, image):
        # Megjeleníti a képet a GUI-n
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def process_image(self, image):
        # Előfeldolgozási lépések
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        unsharp_mask = cv2.addWeighted(gray_image, 1.5, denoised_image, -0.5, 0)
        contrast_image = cv2.equalizeHist(unsharp_mask)
        edged = cv2.Canny(contrast_image, 30, 200)
        #Ez a függvény a kép éleinek (kontúroknak) az azonosítására szolgál
        #Az algoritmus az élek hosszának és görbületének alapján választja ki a legvalószínűbb kontúrt
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            #kontúrok közelítése, meghatározza azokat a téglalapokat vagy négyszögeket, amelyek a rendszámtáblákat reprezentálják
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            # Rendszám kiemelése a képen
            cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)
            mask = np.zeros(gray_image.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(image, image, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped_image = gray_image[topx:bottomx + 1, topy:bottomy + 1]
            return cropped_image
        else:
            return None

    #Az OCR olyan technológia, amely lehetővé teszi a gépek számára, hogy optikai karaktereket (szöveget) ismerjenek fel képekről
    # vagy dokumentumokról
    def recognize_license_plate(self, processed_image):
        # Rendszám kiolvasása OCR segítségével
        if processed_image is not None:
            #Page Segmentation Mode, Ez a beállítás határozza meg, hogy a Tesseract hogyan próbálja meg értelmezni a szöveget
            #--psm 11: Tesseract egy szövegblokkot keres
            license_plate_text = pytesseract.image_to_string(processed_image, config='--psm 11') #8
            if license_plate_text:
                self.result_label.config(text="Rendszám: " + license_plate_text)

    def display_processed_licensePlate_image(self):
        # Előfeldolgozás és rendszám kiolvasás megjelenítése
        processed_image = self.process_image(self.loaded_image)
        self.display_image(processed_image)
        self.recognize_license_plate(processed_image)

# Tkinter ablak inicializálása
root = tk.Tk()
# Rendszamolvaso osztály példányosítása
app = Rendszamolvaso(root)
# Tkinter ablak indítása
root.mainloop()
