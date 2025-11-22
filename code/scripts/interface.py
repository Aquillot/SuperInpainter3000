from tkinter import *
from tkinter import colorchooser, ttk, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
try:
    import torch
except Exception:
    torch = None



from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
import torchvision
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF



from testings.model import UNet
from utils.image import to_img

base_path = "../"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet(3).to(device)

state_dict = torch.load(base_path + "models/unet_300.pt")
# Retirer le préfixe "unet."
new_state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

transform = Compose([
    Resize(256),
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
        self.canvas_width = 500
        self.canvas_height = 400
        self._pil_image = Image.new('RGB', (self.canvas_width, self.canvas_height), self.color_bg)
        self._draw = ImageDraw.Draw(self._pil_image)
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.mask = None
        self.image_to_mask = None

    def paint(self, e):
        if self.old_x and self.old_y:
            w = int(round(self.pen_width))
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=w, fill = self.color_fg, capstyle='round', smooth = True)
            # dessiner aussi sur l'image PIL miroir
            # Pillow line draws inclusive of end points; convert coords to tuple
            self._draw.line([(self.old_x, self.old_y), (e.x, e.y)], fill=self.color_fg, width=w)
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
        self.c = Canvas(self.master, width=500, height=400, bg=self.color_bg)
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
        """
        Retourne un mask binaire issu du canvas PIL miroir.
        - compare la valeur en niveau de gris du pixel (0,0) comme fond
        - retourne shape (1,1,H,W)
        - si `to_torch` et torch est disponible, retourne un `torch.float32` tensor sur `device`
        """
        # récupérer image PIL
        pil = self._pil_image.convert('L')
        arr = np.asarray(pil)
        # background estimated from top-left pixel
        bg = arr[0, 0]
        mask = (arr != bg).astype(np.float32)
        if invert:
            mask = 1.0 - mask

        # shape -> (1,1,H,W)
        mask = mask[np.newaxis, np.newaxis, :, :]

        if to_torch and (torch is not None):
            try:
                t = torch.from_numpy(mask).float().to(device)
                return t
            except Exception:
                return mask
        return mask


    def load_image(self):
        filename = filedialog.askopenfilename(initialdir = "~",
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

        canva_mask = self.get_mask_tensor(to_torch=(torch is not None), device=device)
        canva_mask = canva_mask.float()
        canva_mask = torch.nn.functional.interpolate(canva_mask, size=(256, 256), mode='nearest')
        canva_mask = canva_mask.to(device)
        m = canva_mask
        m = m.float()

        mask_t = 1.0 - m
        if torch is not None and isinstance(mask_t, torch.Tensor):
            mask_t = mask_t.to(device)
        else:
            # convert numpy mask to torch
            mask_t = torch.from_numpy(mask_t).float().to(device)

        # Build masked image: here mask==1 means kept pixels
        masked_img = img_t * mask_t

        net_input = torch.cat([masked_img, mask_t], dim=1)

        model.eval()
        with torch.no_grad():
            reconstructed = model(net_input)

        # Denormalize for display (assumes Normalize(mean=0.5,std=0.5))
        def denorm(t):
            return (t * 0.5 + 0.5).clamp(0, 1)

        img_disp = denorm(img_t[0]).permute(1, 2, 0).cpu().numpy()
        masked_disp = denorm(masked_img[0]).permute(1, 2, 0).cpu().numpy()
        recon_disp = denorm(reconstructed[0]).permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img_disp)
        ax[0].set_title("Image originale")

        ax[1].imshow(masked_disp)
        ax[1].set_title("Image masquée")

        ax[2].imshow(recon_disp)
        ax[2].set_title("Image générée")

        for a in ax:
            a.axis("off")
        plt.show()




win = Tk()
win.title("Paint App")
main(win)
win.mainloop()