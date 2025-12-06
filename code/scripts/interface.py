from tkinter import *
from tkinter import colorchooser, ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np

from testings.model import PenNET, InpaintGenerator

try:
    import torch
except Exception:
    torch = None

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from pennet import InpaintGeneratorPennet
from testings.model import UNet
from testings.utils import to_img

base_path = "../"
models_dir = base_path + "models/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_transform(size):
    return Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

default_input_size = 128
transform = build_transform(default_input_size)


class ImprovedMaskingApp:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'Black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.pen_width = 15
        
        # Canvas dimensions
        self.canvas_width = 128
        self.canvas_height = 128
        
        # Image chargée (PIL Image RGB)
        self.loaded_image = None
        # Masque à la résolution de l'image originale (PIL Image 'L')
        self.mask_image = None
        
        # Variables pour l'affichage
        self.canvas_image_id = None
        self.display_tk_image = None
        self.view_scale = 1.0
        self.view_x0 = 0
        self.view_y0 = 0
        
        # Modèle sélectionné
        self.selected_model_path = None
        self.model = None
        self.model_type = None
        self.input_size = default_input_size
        
        self.setup_ui()
        self.setup_bindings()

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame de contrôle à gauche
        self.controls = Frame(self.master, padx=5, pady=5)
        self.controls.pack(side="left", fill="y")
        
        # Titre
        title = Label(self.controls, text='Adobe3000 Premium', font='Georgia 14 bold')
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Slider pour la largeur du pinceau
        Label(self.controls, text='Taille Pinceau', font='Georgia 12').grid(row=1, column=0, columnspan=2)
        self.slider = ttk.Scale(self.controls, from_=5, to=100, command=self.change_pen_width, orient='vertical', length=150)
        self.slider.set(self.pen_width)
        self.slider.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Label affichant la taille actuelle
        self.pen_label = Label(self.controls, text=f'{self.pen_width} px', font='Georgia 10')
        self.pen_label.grid(row=3, column=0, columnspan=2)
        
        # Boutons principaux
        btn_config = {'width': 18, 'pady': 5}
        
        Button(self.controls, text='📁 Charger Image', command=self.load_image, **btn_config).grid(row=4, column=0, columnspan=2, pady=(15, 5))
        Button(self.controls, text='🤖 Sélectionner Modèle', command=self.select_model_dialog, **btn_config).grid(row=5, column=0, columnspan=2, pady=5)
        Button(self.controls, text='✨ Appliquer Masque', command=self.apply_mask, **btn_config).grid(row=6, column=0, columnspan=2, pady=5)
        Button(self.controls, text='🗑️ Effacer Masque', command=self.clear_mask, **btn_config).grid(row=7, column=0, columnspan=2, pady=5)
        Button(self.controls, text='🔄 Réinitialiser Tout', command=self.clear_all, **btn_config).grid(row=8, column=0, columnspan=2, pady=5)
        
        # Info sur le modèle chargé
        Label(self.controls, text='Modèle:', font='Georgia 10').grid(row=9, column=0, columnspan=2, pady=(15, 0))
        self.model_label = Label(self.controls, text='Aucun', font='Georgia 9', fg='red', wraplength=150)
        self.model_label.grid(row=10, column=0, columnspan=2)
        
        # Canvas principal
        self.canvas = Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg=self.color_bg, cursor='crosshair')
        self.canvas.pack(fill=BOTH, expand=True)
        
        # Menu
        self.setup_menu()

    def setup_menu(self):
        """Configure le menu"""
        menu = Menu(self.master)
        self.master.config(menu=menu)
        
        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='Fichier', menu=file_menu)
        file_menu.add_command(label='Charger Image', command=self.load_image)
        file_menu.add_command(label='Sauvegarder Masque', command=self.save_mask)
        file_menu.add_separator()
        file_menu.add_command(label='Quitter', command=self.master.destroy)
        
        edit_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='Édition', menu=edit_menu)
        edit_menu.add_command(label='Couleur Pinceau', command=self.change_fg)
        edit_menu.add_command(label='Couleur Fond', command=self.change_bg)
        edit_menu.add_separator()
        edit_menu.add_command(label='Effacer Masque', command=self.clear_mask)
        edit_menu.add_command(label='Réinitialiser Tout', command=self.clear_all)

    def setup_bindings(self):
        """Configure les événements de la souris"""
        # Dessin avec bouton gauche
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Effacement avec bouton droit
        self.canvas.bind('<Button-3>', self.start_erase)
        self.canvas.bind('<B3-Motion>', self.erase)
        self.canvas.bind('<ButtonRelease-3>', self.stop_draw)
        
        # Redimensionnement
        self.canvas.bind('<Configure>', self.on_canvas_resize)

    def change_pen_width(self, value):
        """Change la largeur du pinceau"""
        try:
            self.pen_width = float(value)
            self.pen_label.config(text=f'{int(self.pen_width)} px')
        except:
            pass

    def canvas_to_image_coords(self, cx, cy):
        """Convertit les coordonnées canvas en coordonnées image"""
        if self.loaded_image is None:
            return None, None
        ix = int(round((cx - self.view_x0) / max(self.view_scale, 1e-6)))
        iy = int(round((cy - self.view_y0) / max(self.view_scale, 1e-6)))
        return ix, iy

    def draw_circle_on_mask(self, cx, cy, radius, fill_value=255):
        """Dessine un cercle sur le masque à la résolution native"""
        if self.mask_image is None:
            return
        
        ix, iy = self.canvas_to_image_coords(cx, cy)
        if ix is None or iy is None:
            return
        
        # Rayon à la résolution de l'image
        img_radius = max(1, int(round(radius / max(self.view_scale, 1e-6))))
        
        draw = ImageDraw.Draw(self.mask_image)
        bbox = [ix - img_radius, iy - img_radius, ix + img_radius, iy + img_radius]
        draw.ellipse(bbox, fill=fill_value)

    def start_draw(self, event):
        """Commence le dessin"""
        self.old_x = event.x
        self.old_y = event.y
        # Dessiner le premier point
        self.draw_point(event.x, event.y, self.color_fg, 255)

    def start_erase(self, event):
        """Commence l'effacement"""
        self.old_x = event.x
        self.old_y = event.y
        self.draw_point(event.x, event.y, self.color_bg, 0)

    def draw(self, event):
        """Dessine en continu"""
        if self.old_x is not None and self.old_y is not None:
            self.draw_line(self.old_x, self.old_y, event.x, event.y, self.color_fg, 255)
        self.old_x = event.x
        self.old_y = event.y

    def erase(self, event):
        """Efface en continu"""
        if self.old_x is not None and self.old_y is not None:
            self.draw_line(self.old_x, self.old_y, event.x, event.y, self.color_bg, 0)
        self.old_x = event.x
        self.old_y = event.y

    def draw_point(self, x, y, color, mask_value):
        """Dessine un point (cercle) sur le canvas et le masque"""
        r = self.pen_width / 2
        # Dessiner sur le canvas
        self.canvas.create_oval(x - r, y - r, x + r, y + r, 
                               fill=color, outline=color, tags='mask_draw')
        # Dessiner sur le masque
        self.draw_circle_on_mask(x, y, r, mask_value)

    def draw_line(self, x1, y1, x2, y2, color, mask_value):
        """Dessine une ligne de cercles entre deux points"""
        # Calculer la distance
        dx = x2 - x1
        dy = y2 - y1
        distance = max(1, int(np.sqrt(dx*dx + dy*dy)))
        
        # Dessiner des cercles le long de la ligne
        steps = max(1, int(distance / (self.pen_width / 4)))  # Overlap pour un trait continu
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = x1 + t * dx
            y = y1 + t * dy
            self.draw_point(x, y, color, mask_value)

    def stop_draw(self, event):
        """Arrête le dessin"""
        self.old_x = None
        self.old_y = None

    def load_image(self):
        """Charge une image"""
        filename = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("Tous les fichiers", "*.*")]
        )
        if not filename:
            return
        
        try:
            # Charger l'image
            self.loaded_image = Image.open(filename).convert('RGB')
            w, h = self.loaded_image.size
            
            # Créer un masque noir à la même résolution
            self.mask_image = Image.new('L', (w, h), 0)
            
            # Afficher l'image
            self.display_image()
            
            print(f"✅ Image chargée: {w}x{h} pixels")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image:\n{e}")

    def display_image(self):
        """Affiche l'image sur le canvas en conservant les proportions"""
        if self.loaded_image is None:
            return
        
        # Obtenir les dimensions actuelles du canvas
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw <= 1 or ch <= 1:
            cw, ch = self.canvas_width, self.canvas_height
        
        w, h = self.loaded_image.size
        
        # Calculer l'échelle pour conserver les proportions
        scale = min(cw / w, ch / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Centrer l'image
        self.view_scale = scale
        self.view_x0 = (cw - new_w) // 2
        self.view_y0 = (ch - new_h) // 2
        
        # Redimensionner pour l'affichage
        display_img = self.loaded_image.copy().resize((new_w, new_h), Image.Resampling.BILINEAR)
        self.display_tk_image = ImageTk.PhotoImage(display_img)
        
        # Afficher sur le canvas
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                self.view_x0, self.view_y0, anchor='nw', image=self.display_tk_image
            )
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.display_tk_image)
            self.canvas.coords(self.canvas_image_id, self.view_x0, self.view_y0)
        
        # Mettre l'image en arrière-plan
        self.canvas.tag_lower(self.canvas_image_id)

    def on_canvas_resize(self, event):
        """Gère le redimensionnement du canvas"""
        if self.loaded_image is not None:
            # Effacer les dessins de masque
            self.canvas.delete('mask_draw')
            # Réafficher l'image
            self.display_image()

    def clear_mask(self):
        """Efface uniquement le masque, conserve l'image"""
        self.canvas.delete('mask_draw')
        if self.loaded_image is not None:
            w, h = self.loaded_image.size
            self.mask_image = Image.new('L', (w, h), 0)
        print("🗑️ Masque effacé")

    def clear_all(self):
        """Réinitialise tout"""
        self.canvas.delete(ALL)
        self.loaded_image = None
        self.mask_image = None
        self.canvas_image_id = None
        self.display_tk_image = None
        print("🔄 Tout réinitialisé")

    def save_mask(self):
        """Sauvegarde le masque"""
        if self.mask_image is None:
            messagebox.showwarning("Attention", "Aucun masque à sauvegarder")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            self.mask_image.save(filename)
            print(f"💾 Masque sauvegardé: {filename}")

    def change_fg(self):
        """Change la couleur du pinceau"""
        color = colorchooser.askcolor(color=self.color_fg)
        if color[1]:
            self.color_fg = color[1]

    def change_bg(self):
        """Change la couleur de fond"""
        color = colorchooser.askcolor(color=self.color_bg)
        if color[1]:
            self.color_bg = color[1]
            self.canvas['bg'] = self.color_bg

    def list_models(self):
        """Liste les modèles disponibles"""
        import os
        if not os.path.isdir(models_dir):
            return []
        return sorted([f for f in os.listdir(models_dir) if f.endswith(('.pth', '.pt'))])

    def select_model_dialog(self):
        """Ouvre une fenêtre de sélection de modèle"""
        models = self.list_models()
        if not models:
            messagebox.showwarning("Attention", f"Aucun modèle trouvé dans {models_dir}")
            return
        
        dialog = Toplevel(self.master)
        dialog.title('Sélection du modèle')
        dialog.geometry('400x200')
        
        Label(dialog, text='Choisissez un modèle:', font='Georgia 12').pack(pady=10)
        
        combo = ttk.Combobox(dialog, values=models, state='readonly', width=40)
        combo.pack(pady=10)
        if models:
            combo.current(0)
        
        def load_selected():
            model_name = combo.get()
            if model_name:
                self.load_model(models_dir + model_name)
                dialog.destroy()
        
        Button(dialog, text='Charger', command=load_selected, width=15).pack(pady=10)

    def load_model(self, path):
        """Charge un modèle avec gestion intelligente du state_dict"""
        # Charger le fichier
        checkpoint = torch.load(path, map_location=device)
        
        # Extraire le vrai state_dict (peut être enveloppé dans différentes clés)
        if isinstance(checkpoint, dict):
            # Chercher le state_dict dans les clés communes
            for key in ['netG', 'generator', 'state_dict', 'model_state_dict', 'gen_state_dict']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"📦 State_dict trouvé dans la clé: '{key}'")
                    break
            else:
                # Si aucune clé connue, utiliser directement le checkpoint
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Nettoyer les préfixes du state_dict
        def clean_state_dict(state):
            new_state = {}
            for k, v in state.items():
                # Retirer les préfixes courants
                nk = k
                for prefix in ['module.', 'unet.', 'generator.', 'model.', 'net.']:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                        break
                new_state[nk] = v
            return new_state
        
        cleaned_state = clean_state_dict(state_dict)
        
        # Essayer de charger avec différentes architectures
        architectures = [
            ("PENNet", lambda: InpaintGeneratorPennet(init_weights=False)),
            ("PenNET/InpaintGenerator", lambda: InpaintGenerator(PenNET(3))),
            ("UNet", lambda: UNet(3))
        ]
        
        for arch_name, model_builder in architectures:
            try:
                print(f"🔄 Tentative de chargement avec {arch_name}...")
                candidate = model_builder().to(device)
                candidate.load_state_dict(cleaned_state, strict=False)
                
                # Vérifier si le chargement a réussi (au moins 50% des paramètres chargés)
                model_keys = set(candidate.state_dict().keys())
                loaded_keys = set(cleaned_state.keys())
                match_ratio = len(model_keys & loaded_keys) / len(model_keys)
                
                if match_ratio > 0.5:
                    self.model = candidate
                    self.model_type = arch_name
                    # Ajuster la taille d'entrée selon le modèle
                    self.input_size = 512
                    # Mettre à jour le transform global
                    global transform
                    transform = build_transform(self.input_size)
                    model_name = path.split('/')[-1]
                    self.selected_model_path = path
                    self.model_label.config(text=f'{model_name}\n({arch_name})', fg='green')
                    print(f"✅ Modèle chargé avec {arch_name} ({match_ratio*100:.1f}% correspondance)")
                    return
                else:
                    print(f"⚠️ {arch_name}: Trop peu de paramètres correspondent ({match_ratio*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ {arch_name} échec: {str(e)[:100]}")
                continue
        
        # Si aucune architecture n'a fonctionné
        messagebox.showerror("Erreur", 
            f"Impossible de charger le modèle.\n"
            f"Aucune architecture compatible trouvée.\n"
            f"Clés disponibles dans le state_dict:\n{list(cleaned_state.keys())[:5]}..."
        )
        print("❌ Échec du chargement avec toutes les architectures")

    def get_mask_tensor(self):
        """Convertit le masque PIL en tensor PyTorch"""
        if self.mask_image is None or torch is None:
            return None
        
        # Redimensionner le masque pour correspondre à l'entrée du modèle
        resized_mask = self.mask_image.resize((self.input_size, self.input_size), Image.Resampling.NEAREST)
        
        # Convertir en array numpy
        mask_array = np.array(resized_mask).astype(np.float32) / 255.0
        
        # Ajouter les dimensions batch et channel: (1, 1, H, W)
        mask_tensor = torch.from_numpy(mask_array[np.newaxis, np.newaxis, :, :]).float().to(device)
        
        return mask_tensor

    def apply_mask(self):
        """Applique le masque avec le modèle"""
        if self.loaded_image is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image")
            return
        
        if self.model is None:
            messagebox.showwarning("Attention", "Chargez d'abord un modèle")
            return
        
        try:
            print("🔄 Application du masque...")
            
            # Transformer l'image (inclut redimensionnement selon le modèle)
            img_tensor = transform(self.loaded_image).unsqueeze(0).to(device)
            
            # Obtenir le masque (inversé pour le modèle)
            mask_tensor = self.get_mask_tensor()
            if mask_tensor is None:
                messagebox.showwarning("Attention", "Pas de masque détecté")
                return
            
            # Inverser le masque (zone à inpainter = 1)
            #mask_tensor = 1.0 - mask_tensor
            
            # Créer l'image masquée
            fill_value = -1.0 if img_tensor.min().item() < 0.0 else 1.0
            masked_img = img_tensor * (1 - mask_tensor) + fill_value * mask_tensor
            
            # Préparer l'entrée du réseau
            if self.model_type == "PENNet":
                # PEN-Net attend 4 canaux: image RGB + masque
                net_input = torch.cat([masked_img, mask_tensor], dim=1)
            else:
                net_input = torch.cat([masked_img, mask_tensor], dim=1)
            
            # Inférence
            self.model.eval()
            with torch.no_grad():
                if self.model_type == "PENNet":
                    output = self.model(net_input, mask_tensor)
                else:
                    try:
                        output = self.model(net_input, mask_tensor)
                    except TypeError:
                        output = self.model(net_input)
            
            # Extraire la reconstruction
            reconstructed = output[1] if isinstance(output, tuple) and len(output) >= 2 else output
            
            # Combiner reconstruction et image originale
            final_result = reconstructed * mask_tensor + img_tensor * (1.0 - mask_tensor)
            
            # Convertir pour affichage
            img_display = to_img(img_tensor)
            masked_display = to_img(masked_img)
            recon_display = to_img(reconstructed)
            final_display = to_img(final_result)
            
            # Afficher les résultats
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(img_display)
            axes[0, 0].set_title("Image Originale", fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(masked_display)
            axes[0, 1].set_title("Image Masquée", fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(np.array(self.mask_image), cmap='gray')
            axes[0, 2].set_title("Masque", fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(recon_display)
            axes[1, 0].set_title("Reconstruction Complète", fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(final_display)
            axes[1, 1].set_title("Résultat Final", fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            # Différence
            img_arr = np.array(img_display) if not isinstance(img_display, np.ndarray) else img_display
            final_arr = np.array(final_display) if not isinstance(final_display, np.ndarray) else final_display
            diff = np.abs(img_arr.astype(float) - final_arr.astype(float))
            axes[1, 2].imshow(diff.astype(np.uint8))
            axes[1, 2].set_title("Différence", fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Masque appliqué avec succès!")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'application du masque:\n{e}")
            print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    root = Tk()
    root.title("Adobe3000 Premium Edition - Masquage Intelligent")
    root.geometry("900x600")
    app = ImprovedMaskingApp(root)
    root.mainloop()