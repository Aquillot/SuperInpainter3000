from tkinter import *
from tkinter import colorchooser, ttk, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from PIL import ImageGrab

from testings.model import PenNET, InpaintGenerator, UNet

try:
    import torch
except Exception:
    torch = None



from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, InterpolationMode
import torchvision
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF


from testings.utils import to_img

base_path = "../"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try :
    model = InpaintGenerator(UNet(3)).to(device)

    state_dict = torch.load(base_path + "models/gen_epoch_18.pth")
    # Retirer le préfixe "unet."
    new_state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
except Exception as e:
    print ("\033[91m")  # Code ANSI pour le rouge
    print("Could not load model:", e, "\n Try to load with UNet architecture.")
    print ("\033[0m")   # Réinitialiser la couleur
    model = UNet(3).to(device)
    state_dict = torch.load(base_path + "models/gen_epoch_18.pth")
    new_state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

target_model_resolution = 128
canva_resolution = 512

transform = Compose([
    Resize(target_model_resolution, interpolation=InterpolationMode.NEAREST),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])




class main:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'Black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.pen_width = 5
        # creer une image PIL miroir du canvas pour extraction de mask
        self.canvas_width = canva_resolution
        self.canvas_height = canva_resolution
        self._pil_image = Image.new("RGBA", (self.canvas_width, self.canvas_height), (255, 255, 255, 0))
        self._draw = ImageDraw.Draw(self._pil_image, "RGBA")
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.mask = None
        self.image_to_mask = None
    def capture_canvas_as_pil(self):
        """
        Capture le Canvas Tkinter entier en une image PIL.
        Retourne une image PIL RGB identique au Canvas affiché.
        """
        # coordonnées absolues du canvas dans l'écran
        self.master.update()  # s'assurer des coordonnées à jour
        x = self.c.winfo_rootx()
        y = self.c.winfo_rooty()
        w = x + self.c.winfo_width()
        h = y + self.c.winfo_height()

        # capture écran → PIL
        return ImageGrab.grab(bbox=(x, y, w, h))


    def paint(self, e):
        if self.old_x is not None and self.old_y is not None:
            w = int(round(self.pen_width))

            # === 1) Dessin AFFICHÉ sur le Canvas (RGB) ===
            self.c.create_line(
                self.old_x, self.old_y, e.x, e.y,
                width=w, fill=self.color_fg,
                capstyle='round', smooth=True
            )

            # === 2) Dessin INVISIBLE dans l'image PIL (ALPHA seulement) ===
            draw_color = (0, 0, 0, 255)  # alpha = 255 pour le mask
            self._draw.line(
                [(self.old_x, self.old_y), (e.x, e.y)],
                fill=draw_color,
                width=w
            )

        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None
    
    def changedW(self, width):
        # ttk.Scale passes the value as a string; convert to float
        try:
            self.pen_width = float(width)
        except Exception:
            # fallback: keep previous value
            pass
    
    def clearcanvas(self):
        self.c.delete(ALL)
        # reset PIL mirror
        self._pil_image = Image.new('RGB', (self.canvas_width, self.canvas_height), self.color_bg)
        self._draw = ImageDraw.Draw(self._pil_image)
    
    def change_fg(self):
        self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]
    
    def change_bg(self):
        self.color_bg = colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg
        # update PIL mirror background
        # create new image preserving drawn content could be complex; here we reset
        self._pil_image = Image.new('RGB', (self.canvas_width, self.canvas_height), self.color_bg)
        self._draw = ImageDraw.Draw(self._pil_image)

    def drawWidgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        textpw = Label(self.controls, text='Pen Width', font='Georgia 16')
        textpw.grid(row=0, column=0)
        self.slider = ttk.Scale(self.controls, from_=5, to=100, command=self.changedW, orient='vertical')
        self.slider.set(self.pen_width)
        self.slider.grid(row=0, column=1)
        self.controls.pack(side="left")
        self.c = Canvas(self.master, width=target_model_resolution, height=target_model_resolution, bg=self.color_bg)
        self.c.pack(fill=BOTH, expand=True)


        apply_btn = Button(self.controls, text='Apply mask', command=self.apply_mask)
        apply_btn.grid(row=1, column=0, columnspan=2, pady=(10,0))

        load_btn = Button(self.controls, text='Load image', command=self.load_image)
        load_btn.grid(row=2, column=0, columnspan=2, pady=(10,0))

        menu = Menu(self.master)
        self.master.config(menu=menu)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Menu', menu=optionmenu)
        optionmenu.add_command(label='Brush Color', command=self.change_fg)
        optionmenu.add_command(label='Background Color', command=self.change_bg)
        optionmenu.add_command(label='Clear Canvas', command=self.clearcanvas)
        optionmenu.add_command(label='Exit', command=self.master.destroy)    
    
    def get_mask_tensor(self, invert=False, to_torch=True, device='cpu'):
        # image RGBA
        rgba = self._pil_image

        # récupérer le canal alpha
        alpha = np.array(rgba.split()[3])  # canal 3 = alpha

        # alpha > 0 = zone où tu as dessiné
        mask = (alpha > 0).astype(np.float32)

        if invert:
            mask = 1.0 - mask

        mask = mask[None, None, :, :]  # (1,1,H,W)

        if to_torch:
            return torch.from_numpy(mask).float().to(device)

        return mask


    def load_image(self):
        filename = filedialog.askopenfilename(initialdir = ".",
                                          title = "Select a File",
                                          filetypes = (("image files", "*.jpg*"),
                                              ("all files", "*.*")))
        if not filename:
            return
        pil = Image.open(filename).convert('RGB')
        self.image_to_mask = pil  # original PIL image (used for model)

        # create a resized copy for display on canvas (fit canvas size)
        disp = pil.copy()
        disp = disp.resize((self.canvas_width, self.canvas_height), resample=Image.BILINEAR)
        self._loaded_tk_image = ImageTk.PhotoImage(disp)

        # if an image already exists on canvas, update it; otherwise create it
        if getattr(self, 'canvas_image_id', None) is None:
            # create image at top-left, then lower it below drawings
            self.canvas_image_id = self.c.create_image(0, 0, anchor='nw', image=self._loaded_tk_image)
            # ensure image is below drawing items
            self.c.tag_lower(self.canvas_image_id)
        else:
            self.c.itemconfig(self.canvas_image_id, image=self._loaded_tk_image)

    def apply_mask(self):
        if self.image_to_mask is None:
            print("No image loaded. Use 'Load image' first.")
            return

        # prepare image tensor using the same transform used at training
        img_t = transform(self.image_to_mask).unsqueeze(0).to(device)  # (1,3,256,256)

        canva_mask = self.get_mask_tensor(to_torch=(torch is not None), device=device , invert=True)
        canva_mask = canva_mask.float()
        canva_mask = torch.nn.functional.interpolate(canva_mask, size=(target_model_resolution, target_model_resolution), mode='nearest')
        canva_mask = canva_mask.to(device)
        m = canva_mask
        m = m.float()

        mask_t = 1.0 - m

        # Build masked image: here mask==1 means kept pixels
        fill_value = -1.0 if img_t.min().item() < 0.0 else 1.0

        # build input: masked image + mask channel
        masked_img = img_t * (1 - mask_t) + fill_value * mask_t  # fill holes

        net_input = torch.cat([masked_img, mask_t], dim=1)

        model.eval()
        with torch.no_grad():
            feats, reconstructed = model(net_input, mask_t)

        cheated = reconstructed * mask_t + img_t * (1.0 - mask_t)

        img_disp = to_img(img_t)
        masked_disp = to_img(masked_img)
        recon_disp = to_img(reconstructed)
        cheated = to_img(cheated)

        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
        ax[0].imshow(img_disp)
        ax[0].set_title("Image originale")

        ax[1].imshow(masked_disp)
        ax[1].set_title("Image masquée")

        ax[2].imshow(recon_disp)
        ax[2].set_title("Image générée")

        # Image triché avec la partie non masquée de l'originale

        ax[3].imshow(cheated)
        ax[3].set_title("Assemblage généré/original")

        for a in ax:
            a.axis("off")
        plt.show()




win = Tk()
win.title("Adobe3000 Premium edition 0.1$/s")
main(win)
win.mainloop()